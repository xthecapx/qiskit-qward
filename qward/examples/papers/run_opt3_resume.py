"""Resume opt3 campaign from job 33 onward. Skips M3-4, increases timeout."""
import subprocess
import sys
import time

TOKEN = "xOqM75Lz6i83FDYO070132WcBZhG5f8c4Qfg-fC7l_zX"
CHANNEL = "ibm_cloud"
INSTANCE = "thecap"

GROVER_SCRIPT = "qward/examples/papers/grover/grover_ibm.py"
QFT_SCRIPT = "qward/examples/papers/qft/qft_ibm.py"

jobs = []

# Remaining Grover 3q: M3-4 x2 (skip), S3-1 x2, SYM-1 x2, SYM-2 x1
# Skip M3-4 (too slow), replace with extra S3-1 and SYM-2
for _ in range(3):
    jobs.append(("GROVER", "S3-1", GROVER_SCRIPT))
for _ in range(2):
    jobs.append(("GROVER", "SYM-1", GROVER_SCRIPT))
for _ in range(2):
    jobs.append(("GROVER", "SYM-2", GROVER_SCRIPT))

# All QFT 2q: SR2 x 17
for _ in range(17):
    jobs.append(("QFT", "SR2", QFT_SCRIPT))

# All QFT 3q: SR3 x 19 + SP3-P2 x 18
for _ in range(19):
    jobs.append(("QFT", "SR3", QFT_SCRIPT))
for _ in range(18):
    jobs.append(("QFT", "SP3-P2", QFT_SCRIPT))

print(f"Remaining jobs: {len(jobs)}")
print(f"  Grover 3q remaining: 7")
print(f"  QFT 2q: 17")
print(f"  QFT 3q: 37")
print("=" * 60)
sys.stdout.flush()

success = 0
failed = 0
start_all = time.time()

for i, (algo, config, script) in enumerate(jobs, 1):
    elapsed_total = time.time() - start_all
    print(f"\n[{i}/{len(jobs)}] {algo} {config} (elapsed: {elapsed_total:.0f}s)")
    sys.stdout.flush()
    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, script, "--config", config, "--opt-levels", "3",
             "--channel", CHANNEL, "--token", TOKEN, "--instance", INSTANCE],
            capture_output=True, text=True, timeout=600
        )
        dt = time.time() - start
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "Success rate:" in line and "Mean" not in line:
                    print(f"  OK ({dt:.0f}s) - {line.strip()}")
                    break
            success += 1
        else:
            print(f"  FAILED ({dt:.0f}s)")
            err_lines = result.stderr.strip().split("\n")[-3:]
            for l in err_lines:
                print(f"    {l}")
            failed += 1
    except subprocess.TimeoutExpired:
        dt = time.time() - start
        print(f"  TIMEOUT ({dt:.0f}s) - skipping")
        failed += 1
    sys.stdout.flush()

total_time = time.time() - start_all
print("\n" + "=" * 60)
print(f"DONE: {success} succeeded, {failed} failed, {total_time:.0f}s total")
sys.stdout.flush()
