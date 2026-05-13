"""Bulk opt3 campaign: 92 jobs for 2q and 3q Grover + QFT."""

import subprocess
import sys
import time

TOKEN = "xOqM75Lz6i83FDYO070132WcBZhG5f8c4Qfg-fC7l_zX"
CHANNEL = "ibm_cloud"
INSTANCE = "thecap"

GROVER_SCRIPT = "qward/examples/papers/grover/grover_ibm.py"
QFT_SCRIPT = "qward/examples/papers/qft/qft_ibm.py"

# Build job list: (algo, config, script, repeats)
jobs = []

# Grover 2q: 4 configs x 4 repeats = 16
for config in ["S2-00", "S2-1", "S2-10", "S2-11"]:
    for _ in range(4):
        jobs.append(("GROVER", config, GROVER_SCRIPT))

# Grover 3q: 12 configs x ~2 each = 22 (distribute evenly)
grover_3q = [
    "ASYM-1",
    "ASYM-2",
    "H3-0",
    "H3-1",
    "H3-2",
    "H3-3",
    "M3-1",
    "M3-2",
    "M3-4",
    "S3-1",
    "SYM-1",
    "SYM-2",
]
# 22 jobs across 12 configs: 10 configs x 2 + 2 configs x 1 = 22
for i, config in enumerate(grover_3q):
    repeats = 2 if i < 10 else 1
    for _ in range(repeats):
        jobs.append(("GROVER", config, GROVER_SCRIPT))

# QFT 2q: 1 config x 17 repeats = 17
for _ in range(17):
    jobs.append(("QFT", "SR2", QFT_SCRIPT))

# QFT 3q: 2 configs, 37 total -> SR3 x 19 + SP3-P2 x 18 = 37
for _ in range(19):
    jobs.append(("QFT", "SR3", QFT_SCRIPT))
for _ in range(18):
    jobs.append(("QFT", "SP3-P2", QFT_SCRIPT))

print(f"Total jobs: {len(jobs)}")
print(f"  Grover 2q: 16")
print(f"  Grover 3q: 22")
print(f"  QFT 2q:    17")
print(f"  QFT 3q:    37")
print("=" * 60)

success = 0
failed = 0
start_all = time.time()

for i, (algo, config, script) in enumerate(jobs, 1):
    elapsed_total = time.time() - start_all
    print(f"\n[{i}/{len(jobs)}] {algo} {config} (elapsed: {elapsed_total:.0f}s)")
    start = time.time()
    result = subprocess.run(
        [
            sys.executable,
            script,
            "--config",
            config,
            "--opt-levels",
            "3",
            "--channel",
            CHANNEL,
            "--token",
            TOKEN,
            "--instance",
            INSTANCE,
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    dt = time.time() - start
    if result.returncode == 0:
        # Extract success rate from output
        for line in result.stdout.split("\n"):
            if "Success rate:" in line and "Mean" not in line:
                print(f"  OK ({dt:.0f}s) - {line.strip()}")
                break
        success += 1
    else:
        print(f"  FAILED ({dt:.0f}s)")
        # Print last few lines of error
        err_lines = result.stderr.strip().split("\n")[-3:]
        for l in err_lines:
            print(f"    {l}")
        failed += 1

total_time = time.time() - start_all
print("\n" + "=" * 60)
print(f"DONE: {success} succeeded, {failed} failed, {total_time:.0f}s total")
