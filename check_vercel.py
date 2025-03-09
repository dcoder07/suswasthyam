"""
A simple script to check what's available in the Vercel environment.
"""

import sys
import platform
import os
import json

# Store environment information
info = {
    "python_version": sys.version,
    "platform": platform.platform(),
    "python_path": sys.executable,
    "cwd": os.getcwd(),
    "sys_path": sys.path,
    "env_vars": {k: v for k, v in os.environ.items() if k.startswith(("PYTHON", "VERCEL"))},
    "available_modules": []
}

# Check for key modules
modules_to_check = [
    "flask", "numpy", "pandas", "matplotlib", "sklearn", 
    "tensorflow", "joblib", "PIL"
]

for module in modules_to_check:
    try:
        __import__(module)
        info["available_modules"].append(module)
    except ImportError:
        pass

# Output the information
print(json.dumps(info, indent=2))

# Write to file for debugging
with open("vercel_env_info.json", "w") as f:
    json.dump(info, f, indent=2)

print("\nEnvironment check complete. Results saved to vercel_env_info.json") 