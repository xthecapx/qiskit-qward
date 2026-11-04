"""Offline driver tests for bounded Batch submission and pilot repeats."""

import json
import os
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from qward.examples.papers import campaign_matrix as M, campaign_submit as S


class Runtime:
    """Each job finishes at a given clock tick. The active set includes queued jobs."""

    def __init__(self, root, monkeypatch):
        self.paths = {}
        self.end = {}
        self.events = []
        self.tick = 0
        self.peak = 0
        for key in M.RUNNERS:
            path = root / f"{key}-w0-r2_manifest.json"
            self.paths[key] = path
            self.save(key, "prepared")
        monkeypatch.setattr(S, "load_runner", lambda key: SimpleNamespace(campaign=lambda: self))
        monkeypatch.setattr(S.time, "sleep", self.advance)

    def save(self, key, status):
        sent = status != "prepared"
        self.paths[key].write_text(
            json.dumps(
                {
                    "algorithm": key,
                    "backend_name": "fake_fez",
                    "campaign_id": "pilot-r2",
                    "window": 0,
                    "status": status,
                    "batch_id": key if sent else None,
                    "jobs_planned": 1,
                    "items": [{"runs": 1, "job_ids": [key] if sent else []}],
                }
            ),
            encoding="utf-8",
        )
        return json.loads(self.paths[key].read_text(encoding="utf-8"))

    def backend(self, name):
        return name

    def submit_manifest(self, path, backend):
        key = json.loads(path.read_text(encoding="utf-8"))["algorithm"]
        assert key not in self.end
        self.events.append(("submit", key, self.tick))
        self.end[key] = self.tick + (4 if key == "ghz" else 1)
        self.peak = max(self.peak, sum(t > self.tick for t in self.end.values()))
        return self.save(key, "submitted")

    def advance(self, _):
        self.tick += 1

    def job(self, key):
        return SimpleNamespace(in_final_state=lambda: self.tick >= self.end[key])

    def recover_manifest(self, path, service):
        key = json.loads(path.read_text(encoding="utf-8"))["algorithm"]
        assert self.tick >= self.end[key]
        self.events.append(("recover", key, self.tick))
        return self.save(key, "completed")


@pytest.mark.parametrize("limit", [1, 2, 3])
def test_bound_and_refill_as_soon_as_a_batch_finishes(tmp_path, monkeypatch, limit):
    service = Runtime(tmp_path, monkeypatch)
    S.submit_window(service.paths, limit, service)
    assert service.peak == limit
    assert [key for event, key, _ in service.events if event == "submit"] == list(M.RUNNERS)
    assert all(
        json.loads(p.read_text(encoding="utf-8"))["status"] == "completed"
        for p in service.paths.values()
    )
    if limit == 2:
        assert ("submit", "vtp", 1) in service.events  # GHZ is still queued.


def test_resume_counts_existing_batches_before_opening_new_ones(tmp_path, monkeypatch):
    service = Runtime(tmp_path, monkeypatch)
    service.save("rv", "completed")
    for key in ("vtp", "shor"):
        service.save(key, "submitted")
        service.end[key] = 2
    S.submit_window(service.paths, 1, service)
    starts = [(key, tick) for event, key, tick in service.events if event == "submit"]
    assert len(starts) == 5
    assert all(tick >= 2 and key not in ("rv", "vtp", "shor") for key, tick in starts)
    service.events.clear()
    S.submit_window(service.paths, 2, service)
    assert not service.events


def test_unknown_partial_batch_blocks_new_submissions(tmp_path, monkeypatch):
    service = Runtime(tmp_path, monkeypatch)
    manifest = service.save("rv", "submitting")
    manifest["items"][0]["job_ids"] = []
    service.paths["rv"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="partial submission"):
        S.submit_window(service.paths, 2, service)
    assert not service.events


def test_failed_open_survives_restart_without_duplicating_other_batch(tmp_path, monkeypatch):
    service = Runtime(tmp_path, monkeypatch)
    submit = service.submit_manifest

    def refuse_second(path, backend):
        if path == service.paths["bv"]:
            raise RuntimeError("Batch refused")
        return submit(path, backend)

    monkeypatch.setattr(service, "submit_manifest", refuse_second)
    with pytest.raises(RuntimeError, match="Batch refused"):
        S.submit_window(service.paths, 2, service)
    assert json.loads(service.paths["bv"].read_text(encoding="utf-8"))["status"] == "prepared"
    monkeypatch.setattr(service, "submit_manifest", submit)
    S.submit_window(service.paths, 2, service)
    assert service.peak == 2
    assert len([e for e in service.events if e[0] == "submit"]) == 8


def test_repeat_gets_own_budget_and_preserves_original(tmp_path):
    jobs = 0
    for key in M.RUNNERS:
        pilot = M.plan_path(key, 0, tmp_path)
        pilot.parent.mkdir(parents=True)
        original = M.plan_path(key, 0).read_bytes()
        pilot.write_bytes(original)
        path = M.write_pilot_repeat(key, 2, tmp_path)
        repeat = json.loads(path.read_text(encoding="utf-8"))
        assert path.stem == f"{key}-w0-r2" and repeat["window"] == 0
        assert repeat.pop("campaign_id") == M.CAMPAIGN_ID + "-pilot-r2"
        assert {**repeat, "campaign_id": M.CAMPAIGN_ID} == json.loads(original)
        assert pilot.read_bytes() == original
        assert M.write_pilot_repeat(key, 2, tmp_path) == path
        jobs += repeat["planned_executions"]
    assert jobs == 30


@pytest.mark.parametrize(
    "option,expected", [([], 2), (["--max-batches", "1"], 1), (["--max-batches", "3"], 3)]
)
def test_shell_passes_limit_and_exact_repeat_manifests(tmp_path, option, expected):
    papers = tmp_path / "qward/examples/papers"
    papers.mkdir(parents=True)
    driver = papers / "run_campaign.sh"
    shutil.copy2(M.PAPERS / driver.name, driver)
    manifests = []
    for key, runner in M.RUNNERS.items():
        directory = papers / runner["dir"] / "plans"
        directory.mkdir(parents=True)
        for label in ("0", "0-r2"):
            path = directory / f"{key}-w{label}_manifest.json"
            path.touch()
        manifests.append(str(path))
    # Intercept the uv boundary; the final exec arguments show what would run.
    uv = tmp_path / "uv"
    uv.write_text(
        '#!/bin/sh\nif [ "$2" = python ]; then cat >/dev/null; fi\nprintf "%s\\n" "$@"\n',
        encoding="utf-8",
    )
    uv.chmod(0o755)
    proc = subprocess.run(
        [str(driver), "submit", "all", "0-r2", *option],
        input="y\n",
        text=True,
        capture_output=True,
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
        check=True,
    )
    assert proc.stdout.splitlines()[-13:] == [
        "run",
        "-m",
        "qward.examples.papers.campaign_submit",
        "--max-batches",
        str(expected),
        *manifests,
    ]
