import os
import subprocess

cmd = 'pip install -r requirements.txt'
p1 = subprocess.call(cmd, shell=True)

cmd = 'pip install git+https://github.com/openai/CLIP.git'
p1 = subprocess.call(cmd, shell=True)