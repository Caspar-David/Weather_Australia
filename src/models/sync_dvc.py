import subprocess
import sys
from datetime import datetime

def run(cmd):
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Command failed: {cmd}")
        sys.exit(result.returncode)

# Step 1: Commit stage-outputs
run("dvc commit --force")

# Step 2: Git add DVC-tracked files
run("git add dvc.lock")

# Step 3: Git commit
commit_message = f"Update data/model via Airflow at {datetime.utcnow().isoformat()}"

try:
    run(f'git commit -m "{commit_message}"')
except SystemExit as e:
    # Git exits with code 1 if there's nothing to commit – das ist okay
    print("Nothing to commit. Probably no file has changed.")
    sys.exit(0)

# Push to Git + DVC remote
run("git push origin Caspar")
run(f"dvc push -r {remote}")
