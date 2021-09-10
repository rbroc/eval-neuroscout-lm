import glob
import os

fs = glob.glob('outputs/narratives/*')
for f in fs:
    if 'google' in f or 'facebook' in f:
        os.remove(f)