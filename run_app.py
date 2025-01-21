import threading
import subprocess
import uvicorn

def run_fastapi():
    uvicorn.run("app:app", host="0.0.0.0", port=8000)

def run_streamlit():
    subprocess.run(["streamlit", "run", "demo_web.py"])

fastapi_thread = threading.Thread(target=run_fastapi)
fastapi_thread.daemon = True
fastapi_thread.start()

run_streamlit()