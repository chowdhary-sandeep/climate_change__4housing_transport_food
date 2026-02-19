"""Kill labeling processes once checkpoint_0012.json is saved."""
import os, time, psutil

CHK = "E:/climate_change__4housing_transport_food/paper4data/checkpoints/checkpoint_0012.json"
TARGET_PIDS = [20476, 35972]

print("Watching for checkpoint_0012.json ...", flush=True)
while not os.path.exists(CHK):
    time.sleep(10)

print("checkpoint_0012.json found! Terminating labeling processes ...", flush=True)
for pid in TARGET_PIDS:
    try:
        p = psutil.Process(pid)
        p.terminate()
        print(f"  Terminated PID {pid}", flush=True)
    except psutil.NoSuchProcess:
        print(f"  PID {pid} already gone", flush=True)
    except Exception as e:
        print(f"  Error killing PID {pid}: {e}", flush=True)

print("Done.", flush=True)
