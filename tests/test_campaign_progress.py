"""Progress diagnostics must not change operations or leave heartbeat threads."""

from threading import Event, enumerate as threads

import pytest

from qward.examples.papers import campaign_progress as progress


def test_disabled_progress_does_not_start_a_thread(monkeypatch, capsys):
    """Quiet mode must have no output or background worker."""
    monkeypatch.delenv("CAMPAIGN_PROGRESS", raising=False)

    def unexpected_thread(**_kwargs):
        pytest.fail("Disabled progress must not start a thread")

    monkeypatch.setattr(progress, "Thread", unexpected_thread)
    with progress.stage("quiet"):
        progress.report("hidden")
    assert capsys.readouterr() == ("", "")


def test_heartbeat_stops_on_completion(monkeypatch):
    """A slow operation emits a heartbeat and joins its worker on exit."""
    monkeypatch.setenv("CAMPAIGN_PROGRESS", "1")
    emitted = Event()
    messages = []

    def capture(message):
        messages.append(message)
        if "still waiting" in message:
            emitted.set()

    monkeypatch.setattr(progress, "report", capture)
    with progress.stage("slow metric", interval=0.001):
        assert emitted.wait(timeout=2), "Heartbeat did not arrive"
    assert "completed" in messages[-1]
    assert not any(thread.name == "campaign-progress" for thread in threads())


def test_failure_propagates_and_logs_only_to_stderr(monkeypatch, capsys):
    """Failure remains visible to callers and cannot corrupt JSON on stdout."""
    monkeypatch.setenv("CAMPAIGN_PROGRESS", "1")
    with pytest.raises(ValueError, match="original failure"):
        with progress.stage("prepare GHZ8"):
            raise ValueError("original failure")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "UTC] prepare GHZ8: started" in captured.err
    assert "prepare GHZ8: failed/interrupted" in captured.err
    assert not any(thread.name == "campaign-progress" for thread in threads())
