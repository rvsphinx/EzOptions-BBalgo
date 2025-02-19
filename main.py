import subprocess
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def install_requirements():
    logging.info("Checking and installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "-r", "requirements.txt"])
        logging.info("Successfully installed requirements.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error installing requirements: {e}")
        sys.exit(1)

def check_git():
    try:
        subprocess.check_call(["git", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Verify if current directory is a git repository
        subprocess.check_call(["git", "rev-parse", "--git-dir"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        logging.error("Not a valid git repository. Please ensure you're in the correct directory.")
        return False
    except FileNotFoundError:
        logging.error("Git is not installed. Please install Git to update the project.")
        return False

def update_project():
    if not check_git():
        sys.exit(1)
    
    logging.info("Updating project...")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # First, try to fix any potential repository issues
            subprocess.check_call(["git", "config", "--global", "--add", "safe.directory", os.getcwd()])
            
            # Attempt to reset any potential conflicts
            subprocess.check_call(["git", "reset", "--hard", "HEAD"])
            
            # Pull changes
            subprocess.check_call(["git", "pull"])
            logging.info("Successfully updated project.")
            return
        except subprocess.CalledProcessError as e:
            if attempt < max_retries - 1:
                logging.warning(f"Update attempt {attempt + 1} failed, retrying...")
                continue
            logging.error(f"Error updating project (Git error {e.returncode}): {e}")
            logging.error("Please try manually running: git config --global --add safe.directory <your_project_path>")
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
