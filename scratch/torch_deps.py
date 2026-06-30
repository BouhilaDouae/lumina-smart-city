import os
import ctypes
import pefile

path = os.path.join(r'C:\Users\HP\Downloads\Lumina-Smart-Grid-GNN\.venv\Lib\site-packages\torch\lib', 'c10.dll')
print('c10 exists', os.path.exists(path))
pe = pefile.PE(path)
imports = [entry.dll.decode('utf-8', errors='ignore') for entry in pe.DIRECTORY_ENTRY_IMPORT]
print('Imports count', len(imports))
for dll in imports:
    print('IMPORT', dll)

print('\n--- Checking loads ---')
for dll in imports:
    try:
        ctypes.WinDLL(dll)
        print('OK', dll)
    except Exception as e:
        print('FAIL', dll, e)
