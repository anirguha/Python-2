
# copy_git_folder.py
import subprocess
from pathlib import Path
import shutil
import argparse

def copy_git_folder(source_folder: str, target_folder: str):
    """
    Efficiently copies 'Helper Functions' folder from GitHub repo
    into the current working directory using Git sparse checkout.
    """
    repo_url = "https://github.com/anirguha/Python-2.git"
    dest = Path("/content/Python-2")  # temporary repo clone

    # Clean up if exists
    if dest.exists():
        shutil.rmtree(dest)

    print("🚀 Initializing sparse clone...")
    subprocess.run(["git", "init", str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "remote", "add", "-f", "origin", repo_url], check=True)
    subprocess.run(["git", "-C", str(dest), "config", "core.sparseCheckout", "true"], check=True)

    sparse_file = dest / ".git" / "info" / "sparse-checkout"
    sparse_file.parent.mkdir(parents=True, exist_ok=True)
    sparse_file.write_text(f"{source_folder}/*\n")

    subprocess.run(["git", "-C", str(dest), "pull", "origin", "master"], check=True)

    src_folder = dest / source_folder
    target_folder = Path(target_folder)

    if target_folder.exists():
        print(f"⚠️ {target_folder} already exists, removing it...")
        shutil.rmtree(target_folder)

    print("📂 Copying folder to working directory...")
    shutil.move(str(src_folder), str(target_folder))

    print("🧹 Cleaning up temporary repo...")
    shutil.rmtree(dest)

    print("✅ Copy complete!")

    return target_folder

if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("--source_folder", type=str)
  p.add_argument("--target_folder", type=str)
  args = p.parse_args()

  copy_git_folder(args.source_folder, args.target_folder)
