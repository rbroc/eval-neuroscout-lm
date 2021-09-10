import glob
import os

def cleanup():
    fs = glob.glob('outputs/narratives/*')
    for f in fs:
        if 'google' in f or 'facebook' in f:
            os.remove(f)
            
if __name__=='__main__':
    cleanup()