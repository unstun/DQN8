"""Helper: parse success rates from table2_kpis_mean.csv for checkpoint SR screening."""
import csv
import sys

if len(sys.argv) < 2:
    print("N/A,N/A,N/A,N/A,N/A,N/A")
    sys.exit(0)

filepath = sys.argv[1]
kpi = {}

try:
    with open(filepath) as f:
        for row in csv.DictReader(f):
            algo = row.get("Algorithm name", row.get("algo", "")).strip().lower()
            sr = row.get("Success rate", "").strip()
            kpi[algo] = sr
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    print("N/A,N/A,N/A,N/A,N/A,N/A")
    sys.exit(0)


def get(key):
    for k, v in kpi.items():
        if key in k:
            return v
    return "N/A"


# Output: cnn-pddqn,cnn-ddqn,cnn-dqn,mlp-pddqn,mlp-ddqn,mlp-dqn
print(f"{get('cnn-pddqn')},{get('cnn-ddqn')},{get('cnn-dqn')},{get('mlp-pddqn')},{get('mlp-ddqn')},{get('mlp-dqn')}")
