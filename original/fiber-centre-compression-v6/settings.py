from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = Path('E:/fiber-full-8000-v5')
PARENT = Path('D:/fiber-compression-multiscroll-v4')
SOURCE_RUN = SOURCE/'runs/multiscroll-full-v5'
OUT = ROOT/'runs/multiscroll-centre-v6'
PHASE_START = 874
INITIAL_SHA = '847aa6c9b153d947a1c0939e7fb91b6f3021b5c38c5a4ed1590034dbfee5411e'
sys.path.insert(0, str(PARENT/'dependencies'))
sys.path.append(str(SOURCE))
