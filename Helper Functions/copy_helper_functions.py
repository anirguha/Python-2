import subprocess
from pathlib import Path
import shutil

def copy_helper_functions():
    """
    Copies 'Important PyTorch Modules' folder from GitHub repository
    into the current working directory (e.g., Colab).
    """
    # Clone the GitHub repository
    repo_url = "https://github.com/anirguha/Python-2"
    subprocess.run(["git", "clone", repo_url], check=True)

    # Source folder (contains spaces, so use Path safely)
    src_folder = Path("Python-2") / "Helper Functions"
    target_folder = Path(".") / src_folder.name  # "./Helper Functions"

    # If target already exists, remove it
    if target_folder.exists():
        print(f"{target_folder} already exists... deleting.")
        shutil.rmtree(target_folder)

    # Move folder using Python instead of mv
    print("Copying folder to working directory ...")
    shutil.move(str(src_folder), ".")

    # Remove the cloned repo safely
    print("Cleaning up cloned repository ...")
    shutil.rmtree("Python-2")

    print("✅ Copy complete!")
