import subprocess
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def install_requirements():
    logging.info("Checking and installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logging.info("Successfully installed requirements.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error installing requirements: {e}")
        sys.exit(1)

def check_git():
    try:
        subprocess.check_call(["git", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        logging.error("Git is not installed. Please install Git to update the project.")
        return False

def update_project():
    if not check_git():
        sys.exit(1)
    
    logging.info("Updating project...")
    try:
        subprocess.check_call(["git", "pull"])
        logging.info("Successfully updated project.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error updating project: {e}")
        sys.exit(1)

def run_ezoptions():
    logging.info("Running ezoptions.py...")
    try:
        subprocess.check_call(["streamlit", "run", "ezoptions.py"])
        logging.info("Successfully ran ezoptions.py using streamlit.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running ezoptions.py: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install_requirements()
    update_project()
    run_ezoptions()
