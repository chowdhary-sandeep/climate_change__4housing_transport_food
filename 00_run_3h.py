"""
Run labeling for 3 hours then terminate automatically.
Usage: python 00_run_3h.py
Logs -> 00_run_full_label_log.txt
"""
import subprocess, sys, os, time

HOURS   = 3
TIMEOUT = HOURS * 3600

base = os.path.dirname(os.path.abspath(__file__))
log  = os.path.join(base, "00_run_full_label_log.txt")

stop_at = time.strftime('%H:%M:%S', time.localtime(time.time() + TIMEOUT))
print(f"Starting labeling (max {HOURS}h, auto-stop at {stop_at})")
print(f"Log -> {log}")

with open(log, "a", encoding="utf-8") as lf:
    proc = subprocess.Popen(
        [sys.executable, "-X", "utf8", "00_run_full_label.py"],
        stdout=lf, stderr=subprocess.STDOUT, cwd=base,
    )

print(f"PID {proc.pid}  |  auto-stop at {stop_at}")

try:
    proc.wait(timeout=TIMEOUT)
    print("Process finished before time limit.")
except subprocess.TimeoutExpired:
    print(f"\n{HOURS}h elapsed — terminating PID {proc.pid} ...")
    proc.terminate()
    try:
        proc.wait(timeout=30)
        print("Terminated cleanly.")
    except subprocess.TimeoutExpired:
        proc.kill()
        print("Force-killed.")
except KeyboardInterrupt:
    print("\nCtrl-C — terminating ...")
    proc.terminate()
    proc.wait(timeout=30)
    print("Stopped.")
