"""
Launch labeling and stop automatically once a target checkpoint file is saved.
Usage: python 00_run_to_ckpt.py [target_checkpoint]   (default: 30)
"""
import sys, os, time, subprocess, datetime

TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 30
CKPT_FILE = f"paper4data/checkpoints/checkpoint_{TARGET:04d}.json"

base = os.path.dirname(os.path.abspath(__file__))
log  = os.path.join(base, "00_run_full_label_log.txt")

print(f"Starting labeling — will stop after checkpoint {TARGET} ({CKPT_FILE})", flush=True)

with open(log, "a", encoding="utf-8") as lf:
    proc = subprocess.Popen(
        [sys.executable, "-X", "utf8", "00_run_full_label.py"],
        stdout=lf, stderr=subprocess.STDOUT, cwd=base,
    )

print(f"PID {proc.pid}  |  watching for {CKPT_FILE}", flush=True)

try:
    while True:
        if os.path.exists(os.path.join(base, CKPT_FILE)):
            print(f"\nCheckpoint {TARGET} detected — terminating PID {proc.pid} ...", flush=True)
            proc.terminate()
            try:
                proc.wait(timeout=30)
                print("Terminated cleanly.", flush=True)
            except subprocess.TimeoutExpired:
                proc.kill()
                print("Force-killed.", flush=True)
            break
        if proc.poll() is not None:
            print("Process exited on its own.", flush=True)
            break
        time.sleep(15)
except KeyboardInterrupt:
    print("\nCtrl-C — terminating ...", flush=True)
    proc.terminate()
    proc.wait(timeout=30)

print(f"Done at {datetime.datetime.now().strftime('%H:%M:%S')}", flush=True)
