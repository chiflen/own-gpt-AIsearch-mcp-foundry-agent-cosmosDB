"""Starts the unified CFTC AI Assistant + Agent server in the background and writes the PID to server.pid."""
import subprocess
import sys
import os

# Use the venv python to run the app
venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.venv', 'Scripts', 'python.exe')
if not os.path.exists(venv_python):
    venv_python = sys.executable

log_out = open('server.log', 'w', encoding='utf-8')
log_err = open('server.log.err', 'w', encoding='utf-8')

# Launch the unified Gradio app (app.py) which runs on http://127.0.0.1:7861
proc = subprocess.Popen(
    [venv_python, 'app.py'],
    cwd=os.path.dirname(os.path.abspath(__file__)),
    stdout=log_out,
    stderr=log_err,
    creationflags=subprocess.CREATE_NO_WINDOW,
)

with open('server.pid', 'w') as f:
    f.write(str(proc.pid))

print(f"Server started with PID {proc.pid}")
print("Logs: server.log / server.log.err")
print("Web app: http://127.0.0.1:7861 (CFTC RAG Q&A + AI Agent tabs)")
