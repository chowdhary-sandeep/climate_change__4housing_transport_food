"""Detached launcher for 00_vLLM_label_full.py on Windows."""
import subprocess, sys, os

base = os.path.dirname(os.path.abspath(__file__))
log  = os.path.join(base, "logs/00_vLLM_label_full_log.txt")

proc = subprocess.Popen(
    [sys.executable, "-X", "utf8", "00_vLLM_label_full.py"],
    stdout=open(log, "a", encoding="utf-8"),
    stderr=subprocess.STDOUT,
    cwd=base,
    creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
)
print(f"Launched PID {proc.pid} -> {log}")
