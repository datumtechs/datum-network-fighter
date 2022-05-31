import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(CURRENT_DIR, '../..')
sys.path.append(BASE_DIR)
pb_dir = os.path.join(BASE_DIR, 'pb')
sys.path.append(pb_dir)
cfg = {}
