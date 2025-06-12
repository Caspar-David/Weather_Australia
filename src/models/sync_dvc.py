import subprocess
from datetime import datetime

def run(cmd):
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, check=False)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)

# DVC commit
run("dvc commit --force")

# Git staging
run("git add dvc.lock")

# staged changes?
has_changes = subprocess.call("git diff --cached --quiet", shell=True)

# commit, if changes exist
if has_changes != 0:
    timestamp = datetime.now().isoformat()
    commit_msg = f"Update data/model via Airflow at {timestamp}"
    run(f"git commit -m \"{commit_msg}\"")
else:
    print("No changes to commit.")

run("dvc push")