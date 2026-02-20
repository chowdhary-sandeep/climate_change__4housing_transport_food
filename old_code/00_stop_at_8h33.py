"""Kill labeling PIDs at 08:33 (3h after 05:33 start)."""
import time, datetime, subprocess

PIDS     = [37356, 50004]
STOP_AT  = datetime.datetime(2026, 2, 19, 8, 33, 5)

print(f"Watchdog running. Will kill PIDs {PIDS} at {STOP_AT.strftime('%H:%M:%S')}", flush=True)

while datetime.datetime.now() < STOP_AT:
    remaining = (STOP_AT - datetime.datetime.now()).total_seconds()
    print(f"  {int(remaining//60)}m remaining...", flush=True)
    time.sleep(min(300, remaining))   # print every 5 min

print(f"Time's up — killing PIDs {PIDS}", flush=True)
for pid in PIDS:
    try:
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
        print(f"  Killed {pid}", flush=True)
    except Exception as e:
        print(f"  Error killing {pid}: {e}", flush=True)

print("Done.", flush=True)
