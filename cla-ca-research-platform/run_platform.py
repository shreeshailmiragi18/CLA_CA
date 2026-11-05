#!/usr/bin/env python3
import subprocess
import sys
import os
import time

def main():
    root_dir = os.path.abspath(os.path.dirname(__file__))
    os.chdir(root_dir)

    print("=== Starting CLA-CA Research Platform ===")

    # 1. Start FastAPI backend
    print("Starting FastAPI backend on http://localhost:8000 ...")
    backend_process = subprocess.Popen(
        ["uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    # 2. Start Frontend dev server
    print("Starting React Frontend on http://localhost:5173 ...")
    frontend_dir = os.path.join(root_dir, "frontend")
    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=frontend_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    print("\n[SUCCESS] Both Backend and Frontend are running!")
    print(" - Backend API: http://localhost:8000")
    print(" - Frontend Dashboard: http://localhost:5173 (or check Vite output below)")
    print("Press Ctrl+C to stop both services.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping services...")
        backend_process.terminate()
        frontend_process.terminate()
        sys.exit(0)

if __name__ == '__main__':
    main()
