import copy

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError as PydanticValidationError

from app.engine import ValidationError, rename_source_columns, review
from app.main import app
from app.models import RenameRequest, ReviewRequest


client = TestClient(app)


def call_rename(payload):
    return rename_source_columns(RenameRequest.model_validate(payload))


def call_review(nodes, export):
    return review(ReviewRequest.model_validate({"nodes": nodes, "export": export}))


def ids(nodes):
    return [n["id"] for n in nodes]


# source -> filter -> projection export, like examples/request.json
GRAPH = {
    "nodes": [
        {
            "id": "users",
            "type": "source",
            "columns": [
                {"name": "uid", "sensitive": False},
                {"name": "email", "sensitive": True},
                {"name": "country", "sensitive": False},
            ],
        },
        {
            "id": "only_cn",
            "type": "filter",
            "input": "users",
            "references": ["country"],
        },
        {
            "id": "export_users",
            "type": "projection",
            "input": "only_cn",
            "columns": [
                {"name": "user_id", "copy": "uid"},
                {"name": "mail", "copy": "email"},
                {"name": "bucket", "constant": "ALL"},
            ],
        },
    ],
    "export": "export_users",
    "source": "users",
    "renames": [
        {"old_name": "uid", "new_name": "user_id_src"},
        {"old_name": "email", "new_name": "mail_src"},
    ],
}


# ----------------------------- happy paths --------------------------------


def test_rename_flows_through_filter_and_projection_copy():
    res = call_rename(GRAPH)
    users, only_cn, export = res["nodes"]

    # Source columns renamed in place, order and sensitive flags preserved.
    assert users == {
        "id": "users",
        "type": "source",
        "columns": [
            {"name": "user_id_src", "sensitive": False},
            {"name": "mail_src", "sensitive": True},
            {"name": "country", "sensitive": False},
        ],
    }
    # The filter's predicate only references the untouched column.
    assert only_cn["references"] == ["country"]
    # copy references follow their input names; output names stay as declared.
    assert export["columns"] == [
        {"name": "user_id", "copy": "user_id_src"},
        {"name": "mail", "copy": "mail_src"},
        {"name": "bucket", "constant": "ALL"},
    ]
    # Ids, edges, node order and column counts never change.
    assert ids(res["nodes"]) == ids(GRAPH["nodes"])
    assert res["export"] == "export_users"
    assert [len(n.get("columns", [])) for n in res["nodes"]] == [3, 0, 3]

    assert res["renamed_columns"] == [
        {"old_name": "uid", "new_name": "user_id_src"},
        {"old_name": "email", "new_name": "mail_src"},
    ]


def test_renamed_graph_round_trips_through_review():
    res = call_rename(GRAPH)
    follow = call_review(res["nodes"], res["export"])
    assert follow == res["review"]

    # Projection output names freeze the rename, so export names do not change.
    assert [c["name"] for c in res["review"]["columns"]] == [
        "user_id",
        "mail",
        "bucket",
    ]
    mail = res["review"]["columns"][1]
    assert mail["blocked"] is True
    assert mail["content_reasons"] == [
        {"node_id": "users", "column": "mail_src"}
    ]
    assert mail["row_reasons"] == []
    assert res["review"]["blocked"] is True


def test_sensitive_flag_and_constant_value_are_untouched():
    res = call_rename(GRAPH)
    users = res["nodes"][0]
    flags = {c["name"]: c["sensitive"] for c in users["columns"]}
    assert flags == {"user_id_src": False, "mail_src": True, "country": False}
    assert res["nodes"][2]["columns"][2] == {"name": "bucket", "constant": "ALL"}


def test_filter_references_are_renamed():
    payload = {
        "nodes": [
            {"id": "s", "type": "source", "columns": [
                {"name": "a", "sensitive": True}, {"name": "b"}]},
            {"id": "f", "type": "filter", "input": "s", "references": ["a", "b"]},
        ],
        "export": "f",
        "source": "s",
        "renames": [{"old_name": "a", "new_name": "x"}],
    }
    res = call_rename(payload)
    assert res["nodes"][1]["references"] == ["x", "b"]
    follow = call_review(res["nodes"], "f")
    assert follow == res["review"]
    # The predicate column stays sensitive: row taint now cites the new name.
    for col in res["review"]["columns"]:
        assert col["row_reasons"] == [{"node_id": "s", "column": "x"}]


def test_filter_export_passes_renamed_columns_through():
    payload = {
        "nodes": [
            {"id": "s", "type": "source", "columns": [{"name": "a"}, {"name": "b"}]},
            {"id": "f", "type": "filter", "input": "s", "references": ["a"]},
        ],
        "export": "f",
        "source": "s",
        "renames": [{"old_name": "a", "new_name": "a2"}],
    }
    res = call_rename(payload)
    # A filter has no outputs of its own: renamed input columns pass through,
    # so the exported column names change.
    assert [c["name"] for c in res["review"]["columns"]] == ["a2", "b"]


def test_rename_travels_multiple_filters_then_stops_at_projection():
    payload = {
        "nodes": [
            {"id": "s", "type": "source", "columns": [{"name": "a", "sensitive": True}]},
            {"id": "f1", "type": "filter", "input": "s", "references": ["a"]},
            {"id": "f2", "type": "filter", "input": "f1", "references": ["a"]},
            {"id": "p", "type": "projection", "input": "f2",
             "columns": [{"name": "out", "copy": "a"}]},
            {"id": "f3", "type": "filter", "input": "p", "references": ["out"]},
        ],
        "export": "f3",
        "source": "s",
        "renames": [{"old_name": "a", "new_name": "a2"}],
    }
    res = call_rename(payload)
    by_id = {n["id"]: n for n in res["nodes"]}
    assert by_id["f1"]["references"] == ["a2"]
    assert by_id["p"]["columns"] == [{"name": "out", "copy": "a2"}]
    # The change stops at the projection output name: f3 still sees "out".
    assert by_id["f3"]["references"] == ["out"]
    assert [c["name"] for c in res["review"]["columns"]] == ["out"]
    assert res["review"]["columns"][0]["row_reasons"] == [
        {"node_id": "s", "column": "a2"}
    ]


def test_join_keys_and_output_references_follow_each_side_input():
    payload = {
        "nodes": [
            {"id": "orders", "type": "source", "columns": [
                {"name": "uid_enc", "sensitive": True}, {"name": "amount"}]},
            {"id": "users", "type": "source", "columns": [
                {"name": "uid"}, {"name": "name"}, {"name": "amount"}]},
            {"id": "j", "type": "join",
             "left_input": "orders", "right_input": "users",
             "left_key": "uid_enc", "right_key": "uid",
             "outputs": [
                 {"alias": "amt", "side": "left", "column": "amount"},
                 {"alias": "uname", "side": "right", "column": "name"},
             ]},
        ],
        "export": "j",
        "source": "orders",
        "renames": [
            {"old_name": "uid_enc", "new_name": "uid2"},
            {"old_name": "amount", "new_name": "amount_l"},
        ],
    }
    res = call_rename(payload)
    by_id = {n["id"]: n for n in res["nodes"]}

    # The other source keeps an identically named column ("amount") untouched.
    users_cols = [c["name"] for c in by_id["users"]["columns"]]
    assert users_cols == ["uid", "name", "amount"]

    join = by_id["j"]
    assert join["left_key"] == "uid2"
    assert join["right_key"] == "uid"  # other side's input mapping is empty
    assert join["outputs"] == [
        {"alias": "amt", "side": "left", "column": "amount_l"},
        {"alias": "uname", "side": "right", "column": "name"},
    ]
    # Aliases never change; the rename stops here.
    assert [c["name"] for c in res["review"]["columns"]] == ["amt", "uname"]
    for col in res["review"]["columns"]:
        assert col["row_reasons"] == [{"node_id": "orders", "column": "uid2"}]


def test_self_join_applies_same_mapping_to_both_sides():
    payload = {
        "nodes": [
            {"id": "s", "type": "source", "columns": [
                {"name": "id", "sensitive": True}, {"name": "pid"}, {"name": "v"}]},
            {"id": "j", "type": "join",
             "left_input": "s", "right_input": "s",
             "left_key": "id", "right_key": "pid",
             "outputs": [
                 {"alias": "lv", "side": "left", "column": "v"},
                 {"alias": "rp", "side": "right", "column": "pid"},
             ]},
        ],
        "export": "j",
        "source": "s",
        "renames": [
            {"old_name": "id", "new_name": "id2"},
            {"old_name": "pid", "new_name": "parent"},
            {"old_name": "v", "new_name": "val"},
        ],
    }
    res = call_rename(payload)
    join = res["nodes"][1]
    assert join["left_key"] == "id2"
    assert join["right_key"] == "parent"
    assert join["outputs"] == [
        {"alias": "lv", "side": "left", "column": "val"},
        {"alias": "rp", "side": "right", "column": "parent"},
    ]
    assert call_review(res["nodes"], "j") == res["review"]


def test_simultaneous_swap_is_allowed():
    payload = {
        "nodes": [
            {"id": "s", "type": "source", "columns": [
                {"name": "a", "sensitive": True}, {"name": "b"}]},
            {"id": "f", "type": "filter", "input": "s", "references": ["a", "b"]},
        ],
        "export": "f",
        "source": "s",
        "renames": [
            {"old_name": "a", "new_name": "b"},
            {"old_name": "b", "new_name": "a"},
        ],
    }
    res = call_rename(payload)
    assert [c["name"] for c in res["nodes"][0]["columns"]] == ["b", "a"]
    assert res["nodes"][1]["references"] == ["b", "a"]
    # Sensitivity (and therefore row taint) follows the original column:
    # column now named "b" is the tainted one.
    by_name = {c["name"]: c for c in res["review"]["columns"]}
    assert by_name["b"]["row_reasons"] == [{"node_id": "s", "column": "b"}]
    assert by_name["a"]["row_reasons"] == [{"node_id": "s", "column": "b"}]
    assert call_review(res["nodes"], "f") == res["review"]


def test_non_export_branches_are_updated_too():
    payload = {
        "nodes": [
            {"id": "s", "type": "source", "columns": [{"name": "a"}, {"name": "b"}]},
            # Shared projection feeding two downstream filters.
            {"id": "p", "type": "projection", "input": "s",
             "columns": [{"name": "a", "copy": "a"}, {"name": "b", "copy": "b"}]},
            {"id": "export_f", "type": "filter", "input": "p",
             "references": ["a"]},
            {"id": "side_f", "type": "filter", "input": "p",
             "references": ["b"]},  # not an ancestor of the export
            {"id": "dead_f", "type": "filter", "input": "s",
             "references": ["b"]},  # unreachable, directly on the source
        ],
        "export": "export_f",
        "source": "s",
        "renames": [{"old_name": "b", "new_name": "b2"}],
    }
    res = call_rename(payload)
    by_id = {n["id"]: n for n in res["nodes"]}
    assert by_id["p"]["columns"][1] == {"name": "b", "copy": "b2"}
    # References through the projection output names stay as-is (name stopped).
    assert by_id["export_f"]["references"] == ["a"]
    assert by_id["side_f"]["references"] == ["b"]
    # The unreachable filter directly on the source still gets its ref renamed.
    assert by_id["dead_f"]["references"] == ["b2"]
    assert call_review(res["nodes"], "export_f") == res["review"]


def test_rename_of_source_unrelated_to_export_keeps_review_identical():
    payload = {
        "nodes": [
            {"id": "s1", "type": "source",
             "columns": [{"name": "x", "sensitive": True}]},
            {"id": "s2", "type": "source",
             "columns": [{"name": "y", "sensitive": True}]},
            {"id": "e", "type": "projection", "input": "s2",
             "columns": [{"name": "y", "copy": "y"}]},
        ],
        "export": "e",
        "source": "s1",
        "renames": [{"old_name": "x", "new_name": "z"}],
    }
    res = call_rename(payload)
    assert res["nodes"][0]["columns"][0]["name"] == "z"
    assert res["nodes"][2] == payload["nodes"][2]
    # The export review references only the untouched other source.
    assert res["review"] == {
        "export": "e",
        "blocked": True,
        "columns": [
            {"name": "y", "blocked": True,
             "content_reasons": [{"node_id": "s2", "column": "y"}],
             "row_reasons": []}
        ],
    }


def test_source_export_renames_exported_columns_in_place():
    payload = {
        "nodes": [
            {"id": "s", "type": "source", "columns": [
                {"name": "a", "sensitive": True}, {"name": "b"}]},
        ],
        "export": "s",
        "source": "s",
        "renames": [{"old_name": "a", "new_name": "a2"}],
    }
    res = call_rename(payload)
    assert [c["name"] for c in res["nodes"][0]["columns"]] == ["a2", "b"]
    assert res["review"]["columns"][0]["content_reasons"] == [
        {"node_id": "s", "column": "a2"}
    ]
    assert res["review"]["blocked"] is True


def test_other_source_same_named_column_is_not_touched_or_remapped():
    # Both sources have a column "x"; a join's output merges them into alias
    # "u", so one export column carries content reasons from both sources.
    # Renaming s1.x must leave s2.x's origin (node_id, column) exactly as is.
    payload = {
        "nodes": [
            {"id": "s1", "type": "source",
             "columns": [{"name": "k", "sensitive": True},
                         {"name": "x", "sensitive": True}]},
            {"id": "s2", "type": "source",
             "columns": [{"name": "k"}, {"name": "x", "sensitive": True}]},
            {"id": "j", "type": "join",
             "left_input": "s1", "right_input": "s2",
             "left_key": "k", "right_key": "k",
             "outputs": [{"alias": "u", "side": "left", "column": "x"}]},
        ],
        "export": "j",
        "source": "s1",
        "renames": [{"old_name": "x", "new_name": "x1"},
                    {"old_name": "k", "new_name": "k1"}],
    }
    res = call_rename(payload)
    by_id = {n["id"]: n for n in res["nodes"]}
    assert by_id["j"]["left_key"] == "k1"
    assert by_id["j"]["right_key"] == "k"  # other source untouched
    assert by_id["j"]["outputs"][0] == {"alias": "u", "side": "left",
                                       "column": "x1"}
    assert [c["name"] for c in by_id["s2"]["columns"]] == ["k", "x"]

    col = res["review"]["columns"][0]
    assert col["content_reasons"] == [{"node_id": "s1", "column": "x1"}]
    # The sensitive join key of the renamed source taints rows under its new
    # name; s2 contributes nothing here.
    assert col["row_reasons"] == [{"node_id": "s1", "column": "k1"}]

    # Now an export column sourced from s2: its reasons must not be remapped.
    payload2 = copy.deepcopy(payload)
    payload2["nodes"][2]["outputs"] = [
        {"alias": "u", "side": "right", "column": "x"}
    ]
    res2 = call_rename(payload2)
    col2 = res2["review"]["columns"][0]
    assert col2["content_reasons"] == [{"node_id": "s2", "column": "x"}]
    assert col2["row_reasons"] == [{"node_id": "s1", "column": "k1"}]
    assert call_review(res2["nodes"], "j") == res2["review"]


def test_nodes_may_be_out_of_order_and_keep_request_order():
    payload = copy.deepcopy(GRAPH)
    payload["nodes"] = [payload["nodes"][2], payload["nodes"][0], payload["nodes"][1]]
    res = call_rename(payload)
    assert ids(res["nodes"]) == ["export_users", "users", "only_cn"]
    assert call_review(res["nodes"], res["export"]) == res["review"]


def test_noop_rename_is_allowed_and_reports_no_changes():
    payload = copy.deepcopy(GRAPH)
    payload["renames"] = [{"old_name": "uid", "new_name": "uid"}]
    res = call_rename(payload)
    assert res["nodes"] == payload["nodes"]
    assert res["renamed_columns"] == []


# ----------------------------- review equivalence -------------------------


def test_blocked_conclusion_and_reasons_match_mapped_original():
    original_review = call_review(GRAPH["nodes"], GRAPH["export"])
    res = call_rename(GRAPH)

    mapping = {"uid": "user_id_src", "email": "mail_src"}
    assert res["review"]["blocked"] == original_review["blocked"]
    for old, new in zip(original_review["columns"], res["review"]["columns"]):
        # Export column names: projection outputs freeze the rename -> same.
        assert new["name"] == old["name"]
        assert new["blocked"] == old["blocked"]
        for key in ("content_reasons", "row_reasons"):
            mapped = [
                {"node_id": r["node_id"],
                 "column": mapping.get(r["column"], r["column"])}
                for r in old[key]
            ]
            assert new[key] == mapped


# ----------------------------- rejections (400) ---------------------------


def _expect_code(payload, code):
    try:
        call_rename(payload)
    except ValidationError as exc:
        assert exc.code == code
        return
    raise AssertionError(f"expected ValidationError {code}")


def test_unknown_source_node_rejected():
    payload = {**GRAPH, "source": "ghost"}
    _expect_code(payload, "RENAME_INVALID_SOURCE")


def test_non_source_node_as_source_rejected():
    payload = {**GRAPH, "source": "only_cn"}
    _expect_code(payload, "RENAME_INVALID_SOURCE")


def test_missing_old_column_rejected():
    payload = copy.deepcopy(GRAPH)
    payload["renames"][0] = {"old_name": "ghost", "new_name": "x"}
    _expect_code(payload, "RENAME_COLUMN_NOT_FOUND")


def test_rename_colliding_with_untouched_column_rejected_without_partial_graph():
    payload = copy.deepcopy(GRAPH)
    payload["renames"] = [{"old_name": "uid", "new_name": "email"}]
    try:
        call_rename(payload)
        raise AssertionError("expected RENAME_CONFLICT")
    except ValidationError as exc:
        assert exc.code == "RENAME_CONFLICT"
        assert exc.node_id == "users"


def test_two_renames_converging_rejected():
    payload = copy.deepcopy(GRAPH)
    payload["renames"] = [
        {"old_name": "uid", "new_name": "x"},
        {"old_name": "email", "new_name": "x"},
    ]
    _expect_code(payload, "RENAME_CONFLICT")


def test_original_graph_is_validated_first_and_rejected_as_before():
    # A bad reference in a branch the export never uses still rejects.
    payload = copy.deepcopy(GRAPH)
    payload["nodes"].append(
        {"id": "bad", "type": "filter", "input": "users",
         "references": ["ghost"]}
    )
    _expect_code(payload, "INVALID_REFERENCE")

    cyclic = copy.deepcopy(GRAPH)
    cyclic["nodes"][1]["input"] = "only_cn"
    _expect_code(cyclic, "CYCLE")

    dup = copy.deepcopy(GRAPH)
    dup["nodes"].append(copy.deepcopy(dup["nodes"][0]))
    _expect_code(dup, "DUPLICATE_NODE")

    bad_export = {**GRAPH, "export": "ghost"}
    _expect_code(bad_export, "INVALID_REFERENCE")


def test_failure_returns_no_partial_graph_over_http():
    payload = copy.deepcopy(GRAPH)
    payload["renames"][0] = {"old_name": "ghost", "new_name": "x"}
    resp = client.post("/api/rename", json=payload)
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "RENAME_COLUMN_NOT_FOUND"
    assert "nodes" not in body and "review" not in body


# ----------------------------- schema (422) -------------------------------


def test_duplicate_old_name_rejected_by_schema():
    payload = copy.deepcopy(GRAPH)
    payload["renames"].append({"old_name": "uid", "new_name": "other"})
    with pytest.raises(PydanticValidationError):
        RenameRequest.model_validate(payload)


def test_schema_rejects_empty_lists_and_names_and_unknown_fields():
    def invalid(payload):
        try:
            RenameRequest.model_validate(payload)
        except PydanticValidationError:
            return True
        return False

    base = copy.deepcopy(GRAPH)
    assert invalid({**base, "renames": []})
    bad = copy.deepcopy(base)
    bad["renames"][0] = {"old_name": "uid", "new_name": ""}
    assert invalid(bad)
    bad = copy.deepcopy(base)
    bad["renames"][0] = {"old_name": "", "new_name": "x"}
    assert invalid(bad)
    assert invalid({**base, "unknown": 1})
    bad = copy.deepcopy(base)
    bad["renames"][0]["what"] = 1
    assert invalid(bad)
    assert invalid({k: v for k, v in base.items() if k != "source"})
    assert invalid({k: v for k, v in base.items() if k != "renames"})


def test_more_than_60_nodes_rejected_by_schema():
    nodes = [{"id": "s", "type": "source",
              "columns": [{"name": "a"}]}]
    nodes += [
        {"id": f"p{i}", "type": "projection",
         "input": "s" if i == 0 else f"p{i - 1}",
         "columns": [{"name": "a", "copy": "a"}]}
        for i in range(60)
    ]
    payload = {
        "nodes": nodes,
        "export": "p59",
        "source": "s",
        "renames": [{"old_name": "a", "new_name": "a2"}],
    }
    with pytest.raises(PydanticValidationError):
        RenameRequest.model_validate(payload)


# ----------------------------- HTTP behavior ------------------------------


def test_http_rename_success_and_followup_review():
    resp = client.post("/api/rename", json=GRAPH)
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"nodes", "export", "review", "renamed_columns"}

    follow = client.post(
        "/api/review", json={"nodes": body["nodes"], "export": body["export"]}
    )
    assert follow.status_code == 200
    assert follow.json() == body["review"]


def test_http_rename_schema_error_returns_422():
    resp = client.post("/api/rename", json={"nodes": [], "export": "e"})
    assert resp.status_code == 422


def test_other_endpoints_still_work_after_rename_addition():
    resp = client.post("/api/review", json={
        "nodes": GRAPH["nodes"], "export": GRAPH["export"]
    })
    assert resp.status_code == 200
    assert resp.json()["blocked"] is True

    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
