import os
import glob
import json
import time

print("=== INSPECTION OF EXPERIMENT ARTIFACTS ===")
exp_csv = "experiments.csv"
if os.path.exists(exp_csv):
    mtime = time.ctime(os.path.getmtime(exp_csv))
    print(f"experiments.csv exists. Size: {os.path.getsize(exp_csv)} bytes, MTime: {mtime}")
    with open(exp_csv, "r", encoding="utf-8") as f:
        print("--- experiments.csv contents ---")
        print(f.read())
else:
    print("experiments.csv does NOT exist locally.")

print("\n=== CHECKING CHECKPOINTS DIRECTORY ===")
if os.path.exists("checkpoints"):
    found_any = False
    for root, dirs, files in os.walk("checkpoints"):
        for f in files:
            found_any = True
            p = os.path.join(root, f)
            print(f"{p} | size: {os.path.getsize(p)} bytes | mtime: {time.ctime(os.path.getmtime(p))}")
    if not found_any:
        print("checkpoints directory exists but is empty.")
else:
    print("checkpoints directory does not exist locally.")

print("\n=== CHECKING RESULTS JSON FILES ===")
json_files = glob.glob("**/*results*.json", recursive=True)
if json_files:
    for jf in json_files:
        print(f"{jf} | size: {os.path.getsize(jf)} bytes | mtime: {time.ctime(os.path.getmtime(jf))}")
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
                status = data.get("status")
                exp_id = data.get("experiment_id")
                ts = data.get("timestamp")
                print(f"  status: {status}, exp_id: {exp_id}, timestamp: {ts}")
                test_perf = data.get("test_performance", {})
                print(f"  macro_f1: {test_perf.get('f1_macro')}, acc: {test_perf.get('accuracy')}")
        except Exception as e:
            print(f"  Error reading json: {e}")
else:
    print("No results*.json files found locally.")
