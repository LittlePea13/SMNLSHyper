import os, subprocess

os.environ['FLASK_APP'] = "flask_attention.py"
os.environ['FLASK_DEBUG'] = "1"
subprocess.call(['flask', 'run'])