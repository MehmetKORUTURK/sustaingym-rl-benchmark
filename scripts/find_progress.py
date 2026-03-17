import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

env = sys.argv[1] if len(sys.argv) > 1 else None
root = _REPO_ROOT / "logs_saferl_train"

search_dirs = []
if env:
    search_dirs.append(root / f"omnisafe_{env}")
    patterns = {"building": "Building", "evcharging": "EVCharging", "cogen": "Cogen"}
    if env in patterns:
        search_dirs.extend(d for d in root.glob(f"*{patterns[env]}*") if d.is_dir())
else:
    search_dirs.append(root)

results = set()
for d in search_dirs:
    if d.is_dir():
        results.update(d.rglob("progress.csv"))

results = sorted(results)
outfile = f"{env or 'all'}_progress_paths.txt"
with open(outfile, "w") as f:
    for p in results:
        f.write(f"{p}\n")

print(f"Found {len(results)} progress.csv files -> saved to {outfile}")