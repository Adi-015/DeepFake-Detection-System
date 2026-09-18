import os
import sys
from pathlib import Path

print("=== Checking PIP_CACHE_DIR in user env ===")
try:
    import winreg
    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment")
    val, _ = winreg.QueryValueEx(key, "PIP_CACHE_DIR")
    print(f"Registry HKCU\\Environment\\PIP_CACHE_DIR = {val}")
except Exception as e:
    print(f"Failed to read HKCU\\Environment\\PIP_CACHE_DIR: {e}")

local_app_data = os.environ.get("LOCALAPPDATA", r"C:\Users\admin\AppData\Local")
c_pip_cache = Path(local_app_data) / "pip" / "cache"
print(f"\n=== C: pip cache at {c_pip_cache} ===")
if c_pip_cache.exists():
    total_size = sum(f.stat().st_size for f in c_pip_cache.rglob('*') if f.is_file())
    file_count = sum(1 for f in c_pip_cache.rglob('*') if f.is_file())
    print(f"C: cache exists. Files: {file_count}, Total size: {total_size} bytes ({total_size / (1024*1024):.2f} MB)")
else:
    print("C: cache folder does not exist.")

g_pip_cache = Path(r"G:\deepfake-project\pip-cache")
print(f"\n=== G: pip cache at {g_pip_cache} ===")
if g_pip_cache.exists():
    total_size = sum(f.stat().st_size for f in g_pip_cache.rglob('*') if f.is_file())
    file_count = sum(1 for f in g_pip_cache.rglob('*') if f.is_file())
    print(f"G: cache exists. Files: {file_count}, Total size: {total_size} bytes ({total_size / (1024*1024):.2f} MB)")
else:
    print("G: cache folder does not exist.")
