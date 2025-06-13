import subprocess
from datetime import datetime

def run(cmd):
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, check=False)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)

def configure_git():
    """Setzt Name und Email für Git im Container."""
    run('git config --global user.name "Caspar_Stordeur"') # Change!
    run('git config --global user.email "casparstordeur@gmail.com"')  # Change!

def setup_dvc_remote():
    """Setzt das DVC-Remote auf DagsHub (nutzt .netrc für Auth)."""
    run("dvc remote add -d origin https://dagshub.com/casparstordeur/Weather_Australia.dvc")
    run("dvc remote modify origin --local auth basic")

def main():
    configure_git()

    try:
        run("dvc remote list")
    except subprocess.CalledProcessError:
        setup_dvc_remote()

    # Tracked Files
    tracked_files = [
        "data/processed/X_train.csv",
        "data/processed/X_test.csv",
        "data/processed/y_train.csv",
        "data/processed/y_test.csv",
        "data/processed/xgboost_model.pkl"
    ]
    for file in tracked_files:
        run(f"dvc add {file}")

    run("dvc commit --force")
    run("git add data/processed/*.dvc dvc.lock")

    if subprocess.call("git diff --cached --quiet", shell=True) != 0:
        timestamp = datetime.now().isoformat()
        run(f'git commit -m "Update data/model via Airflow at {timestamp}"')
    else:
        print("No changes to commit.")

    run("dvc push -r origin")

if __name__ == "__main__":
    main()
