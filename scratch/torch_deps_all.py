import os
import ctypes

libdir = os.path.join(r'C:\Users\HP\Downloads\Lumina-Smart-Grid-GNN\.venv\Lib\site-packages\torch\lib')
if not os.path.isdir(libdir):
    raise SystemExit(f'missing libdir: {libdir}')

for filename in sorted(os.listdir(libdir)):
    if filename.lower().endswith('.dll'):
        path = os.path.join(libdir, filename)
        try:
            ctypes.WinDLL(path)
            print('OK', filename)
        except Exception as e:
            print('FAIL', filename, type(e).__name__, e)
