"""``on_session_switch`` (TIM-94, decision 5; TIM-95 decision 4): clear the
session's in-context set on compression, reset or rewind; rebind on every
switch; a plain ``/resume`` clears nothing."""

from conftest import SESSION


def test_compression_clears_the_same_session(make_provider, daemon):
    provider = make_provider()
    provider.on_session_switch(SESSION, parent_session_id=SESSION, reset=False, rewound=False, reason="compression")
    clear = daemon.requests_for("clear")
    assert len(clear) == 1
    assert clear[0].bank == "tim" and clear[0].session == SESSION


def test_reset_clears_the_old_session_and_rebinds(make_provider, daemon):
    provider = make_provider()
    provider.on_session_switch("sess-0002", parent_session_id=SESSION, reset=True)
    assert [r.session for r in daemon.requests_for("clear")] == [SESSION]
    provider.prefetch("tea?")
    assert daemon.requests_for("prefetch")[0].body["session_id"] == "sess-0002"


def test_rewound_clears(make_provider, daemon):
    provider = make_provider()
    provider.on_session_switch(SESSION, parent_session_id=SESSION, rewound=True)
    assert [r.session for r in daemon.requests_for("clear")] == [SESSION]


def test_resume_or_branch_only_rebinds(make_provider, daemon):
    provider = make_provider()
    provider.on_session_switch("sess-0003", parent_session_id=SESSION)
    assert daemon.requests_for("clear") == []
    provider.prefetch("tea?")
    assert daemon.requests_for("prefetch")[0].body["session_id"] == "sess-0003"


def test_clearing_drops_the_pending_recall_id_and_previous_query(make_provider, daemon):
    from conftest import transcript

    provider = make_provider()
    daemon.set_response("prefetch", 200, {"recall_id": "r-before", "text": "x", "injected": ["m1"], "reranked": True})
    provider.prefetch("before compaction", session_id=SESSION)
    provider.on_session_switch(SESSION, parent_session_id=SESSION, reason="compression")
    # The pending recall id went with the in-context set: a turn synced now echoes nothing.
    provider.sync_turn("after", "ok", session_id=SESSION, messages=transcript("after", "ok"))
    assert daemon.requests_for("turns")[0].body["recall_id"] is None
    # And the previous query is gone too.
    provider.prefetch("after", session_id=SESSION)
    assert daemon.requests_for("prefetch")[-1].body.get("previous_query") is None


def test_clear_failure_never_raises_and_still_rebinds(make_provider, daemon):
    provider = make_provider()
    daemon.set_response("clear", 500, {"error": "store"})
    provider.on_session_switch("sess-0002", parent_session_id=SESSION, reset=True)
    url = daemon.go_down()
    provider.on_session_switch("sess-0003", parent_session_id="sess-0002", reset=True)
    assert provider._session_id == "sess-0003"
