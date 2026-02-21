"""Recompute accuracy metrics from updated verification samples."""
import json, sys, os, importlib.util

# Import calc_all_metrics from the file with a numeric-starting name
spec = importlib.util.spec_from_file_location(
    "gpt_verify", os.path.join(os.path.dirname(__file__), "00_GPT_verify_stratified.py")
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
calc_all_metrics = mod.calc_all_metrics

with open('paper4data/00_verification_samples_ckpt.json', encoding='utf-8') as f:
    relabeled = json.load(f)

print("Computing metrics...")
results = calc_all_metrics(relabeled)

print(f"\n=== Summary ===")
s = results['summary']
print(f"Mean accuracy:  {s['mean_accuracy']}")
print(f"Std accuracy:   {s['std_accuracy']}")
print(f"Mean kappa:     {s['mean_kappa']}")
print(f"Total samples:  {s['total_samples']}")

print("\n=== Per-question accuracy (norms) ===")
for qid, m in results['by_task'].get('norms', {}).items():
    print(f"  {qid}: acc={m.get('accuracy')}, kappa={m.get('cohen_kappa')}")

with open('paper4data/00_verification_results_ckpt.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\nSaved to paper4data/00_verification_results_ckpt.json")
