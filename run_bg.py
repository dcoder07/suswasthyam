"""
Run the Flask application in the background.
This script will start the Flask server as a background process.
"""

import subprocess
import os
import sys
import platform

if __name__ == "__main__":
    # Detect the operating system to use the right syntax
    if platform.system() == "Windows":
        # For Windows
        print("Starting Flask server in background (Windows)...")
        # Use pythonw.exe if available to hide console window
        python_exe = "pythonw" if os.path.exists(os.path.join(sys.prefix, 'pythonw.exe')) else "python"
        subprocess.Popen([python_exe, "run_local.py"], 
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        print("Server started! Access at http://127.0.0.1:5000")
        print("To stop the server, you'll need to end the Python process manually.")
        print("You can now use this terminal for Git operations.")
    else:
        # For Unix/Linux/Mac
        print("Starting Flask server in background (Unix)...")
        subprocess.Popen(["nohup", "python", "run_local.py", "&"], 
                        stdout=open("flask.log", "w"),
                        stderr=subprocess.STDOUT)
        print("Server started! Access at http://127.0.0.1:5000")
        print("Check flask.log for server output.")
        print("You can now use this terminal for Git operations.") 