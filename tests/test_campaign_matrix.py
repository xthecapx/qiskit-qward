"""Campaign matrix and the 27 plan files (offline; nothing is prepared or submitted)."""

import json
import re
import subprocess
from pathlib import Path

import pytest
from qiskit_ibm_runtime.fake_provider import FakeFez
from types import SimpleNamespace

from qward.examples.papers import campaign_matrix as M

ROOT = Path(M.PAPERS)


def test_matrix_configs_exist_and_are_unique():
    assert M.validate_matrix() == {}
    assert set(M.MATRIX) == set(M.RUNNERS)


def test_matrix_sizes_match_the_plan():
    sizes = {key: len(configs) for key, configs in M.MATRIX.items()}
    assert sizes == {
        "ghz": 15,
        "bv": 39,
        "vtp": 30,
        "qft": 27,
        "pe": 21,
        "grover": 24,
        "rv": 15,
        "shor": 1,
    }
    assert sum(sizes.values()) * M.RUNS_PER_WINDOW * len(M.WINDOWS) == 1032


def test_known_failure_anchors_and_cuts():
    assert "S5-1" in M.MATRIX["grover"] and "S6-1" not in M.MATRIX["grover"]
    assert M.MATRIX["shor"] == ["SHOR-N15-M3"]
    assert "SR10" not in M.MATRIX["qft"] and "SP12-P8" not in M.MATRIX["qft"]
    assert max(int(re.search(r"BV(\d+)", c).group(1)) for c in M.MATRIX["bv"]) == 14
    assert "RV9-D9" not in M.MATRIX["rv"] and "RV8-D8" in M.MATRIX["rv"]
    assert not any(
        c.startswith("PE-") and int(re.search(r"(\d+)$", c).group(1)) > 8 for c in M.MATRIX["pe"]
    )


def test_plan_files_on_disk_match_the_matrix():
    for key in M.MATRIX:
        for window in (M.PILOT_WINDOW, *M.WINDOWS):
            path = M.plan_path(key, window)
            assert path.exists(), path
            assert json.loads(path.read_text()) == M.plan_for(key, window)


def test_plans_encode_the_agreed_settings():
    plan = M.plan_for("bv", 2)
    assert plan["backend"] == "ibm_fez" and plan["shots"] == 1024
    assert plan["optimization_level"] == 3 and plan["seed_transpiler"] == 11
    assert plan["runs"] == 2 and plan["window"] == 2
    assert plan["planned_executions"] == 78
    assert len(plan["items"]) == 39
    orders = {M.plan_for(k, w)["order_seed"] for k in M.MATRIX for w in M.WINDOWS}
    assert len(orders) == len(M.MATRIX) * len(M.WINDOWS)  # Distinct shuffles.


@pytest.mark.parametrize("key", sorted(M.MATRIX))
def test_every_plan_passes_prepare_plan_validation(key, tmp_path, monkeypatch):
    """Run the real validation path (no records for configs beyond the first)."""
    runner = M.load_runner(key)
    runner.include_quantum_specific = lambda config: False
    runner.get_campaign_output_dir = lambda: tmp_path / "v2"
    plan = M.plan_for(key, 1)
    plan["items"] = plan["items"][:1]  # Keep the test fast; validation is per item.
    plan["planned_executions"] = plan["runs"]
    path = tmp_path / f"{key}-w1.json"
    path.write_text(json.dumps(plan))
    backend = FakeFez()
    backend.refresh = lambda: None
    manifest = runner.campaign().prepare_plan(path, SimpleNamespace(backend=lambda name: backend))
    assert manifest["jobs_planned"] == M.RUNS_PER_WINDOW
    assert manifest["job_order"] and manifest["status"] == "prepared"


def test_full_plan_is_valid_against_budget_and_duplicates(tmp_path):
    for key in M.MATRIX:
        plan = M.plan_for(key, 1)
        configs = [item["config"] for item in plan["items"]]
        assert len(set(configs)) == len(configs)
        assert plan["planned_executions"] == len(configs) * plan["runs"]


def test_driver_paths_match_matrix_paths():
    """run_campaign.sh looks for <dir>/plans/<algo>-w<N>.json; so does the matrix."""
    script = (ROOT / "run_campaign.sh").read_text()
    for key, info in M.RUNNERS.items():
        assert f'"{key}"' in script or f" {key} " in script or key in script
        assert M.plan_path(key, 1) == ROOT / info["dir"] / "plans" / f"{key}-w1.json"
    assert 'pe) echo "phase_estimation"' in script and 'rv) echo "random_volumetric"' in script


def test_driver_status_runs_offline_without_manifests():
    result = subprocess.run(
        [str(ROOT / "run_campaign.sh"), "status", "ghz"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0 and "ALGO" in result.stdout


def test_qpu_time_model_is_monotone_and_bounded():
    assert M.qpu_seconds(10) >= 2
    assert M.qpu_seconds(1000) >= M.qpu_seconds(10)
    assert M.qpu_seconds(5000) < 30  # A 1,024-shot job never models as minutes.


def test_pilot_is_a_small_subset_of_the_matrix_with_its_own_window():
    assert M.PILOT_WINDOW not in M.WINDOWS
    jobs = 0
    for key, configs in M.PILOT.items():
        assert set(configs) <= set(M.MATRIX[key])
        plan = M.plan_for(key, M.PILOT_WINDOW)
        assert plan["window"] == 0 and plan["runs"] == M.PILOT_RUNS
        assert plan["planned_executions"] == len(configs) * M.PILOT_RUNS
        jobs += plan["planned_executions"]
    assert jobs == 30  # ~2 minutes of QPU time
    assert M.PILOT["vtp"] == ["VTP-V2-SINGLE", "VTP-S2-TRIPLE"]  # Static and dynamic.
    assert M.PILOT["shor"] == ["SHOR-N15-M3"]


def test_pilot_and_real_windows_have_separate_run_budgets():
    pilot = M.plan_for("ghz", M.PILOT_WINDOW)
    real = M.plan_for("ghz", 1)
    assert pilot["campaign_id"] == real["campaign_id"] and pilot["window"] != real["window"]
    assert pilot["order_seed"] != real["order_seed"]


@pytest.mark.parametrize("cwd", ["root", "papers", "elsewhere"])
def test_driver_all_works_from_any_directory_and_path_style(cwd, tmp_path):
    """`all` re-invokes the script; it must not depend on a relative $0."""
    papers = ROOT
    directory = {"root": papers.parents[2], "papers": papers, "elsewhere": tmp_path}[cwd]
    absolute = str(papers / "run_campaign.sh")
    commands = [absolute]
    if cwd == "papers":
        commands.append("./run_campaign.sh")
    if cwd == "root":
        commands.append("qward/examples/papers/run_campaign.sh")
    for command in commands:
        result = subprocess.run(
            [command, "recover", "all", "9"],  # Window 9 never has manifests.
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert "No such file or directory" not in result.stderr, (cwd, command, result.stderr)
        assert "no manifest" in result.stderr  # Reached the per-algorithm step.


def test_coin_toss_is_not_part_of_the_campaign():
    assert "coin_toss" not in M.MATRIX and "coin_toss" not in M.RUNNERS
    assert "coin_toss" not in M.PILOT and "coin_toss" in M.EXCLUDED_ALGORITHMS
    for window in M.WINDOWS:
        assert not M.plan_path("ghz", window).with_name(f"coin_toss-w{window}.json").exists()
    assert "coin_toss" not in (ROOT / "run_campaign.sh").read_text()


def test_order_seeds_are_stable_per_algorithm():
    """Removing an algorithm must not change the plans of the others."""
    assert set(M.MATRIX) <= set(M.ORDER_SEED_OFFSET)
    assert len(set(M.ORDER_SEED_OFFSET.values())) == len(M.ORDER_SEED_OFFSET)
    assert M.plan_for("ghz", 0)["order_seed"] == 2 and M.plan_for("shor", 0)["order_seed"] == 7
    assert M.plan_for("vtp", 3)["order_seed"] == 3008


def test_qpu_time_model_reproduces_the_pilot_billing():
    """34 pilot jobs on ibm_fez: 32 billed 2 s (depth <= 440), 2 Shor jobs billed 3 s (depth 4640)."""
    for depth in (4, 14, 44, 77, 135, 140, 177, 438):
        assert M.qpu_seconds(depth) == 2
    assert M.qpu_seconds(4640) == 3
    assert M.qpu_seconds(100000) > M.qpu_seconds(4640)


def test_pilot_estimate_matches_ibm_batch_usage():
    four_job_batch = 4 * M.qpu_seconds(100)
    assert four_job_batch == 8  # The dashboard showed 8 s for every 4-job batch.
