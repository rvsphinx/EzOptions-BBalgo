import subprocess
import sys
import os

def install_requirements():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Successfully installed requirements.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        sys.exit(1)

def update_project():
    try:
        subprocess.check_call(["git", "pull"])
        print("Successfully updated project.")
    except subprocess.CalledProcessError as e:
        print(f"Error updating project: {e}")
        sys.exit(1)

def run_ezoptions():
    try:
        subprocess.check_call(["streamlit", "run", "ezoptions.py"])
        print("Successfully ran ezoptions.py using streamlit.")
    except subprocess.CalledProcessError as e:
        print(f"Error running ezoptions.py: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("Checking and installing requirements...")
    install_requirements()
    
    print("Updating project...")
    update_project()
    
    print("Running ezoptions.py...")
    run_ezoptions()
