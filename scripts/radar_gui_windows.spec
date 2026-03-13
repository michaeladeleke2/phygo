# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs
import sys
import site
import os
from pathlib import Path

block_cipher = None

# Get repo root (phygo/) - we're running from scripts/ folder
current_dir = Path(os.getcwd())  # Should be C:\Users\madeleke\phygo\scripts
repo_root = current_dir.parent   # Go up to C:\Users\madeleke\phygo

print(f"Current directory: {current_dir}")
print(f"Repo root: {repo_root}")

# Collect Infineon SDK data and DLLs
infineon_datas = collect_data_files('ifxradarsdk')
infineon_binaries = collect_dynamic_libs('ifxradarsdk')

# Manual DLL collection for Infineon SDK (backup method)
site_packages = Path(site.getsitepackages()[0])
sdk_lib_path = site_packages / 'ifxradarsdk' / 'lib'

if sdk_lib_path.exists():
    print(f"Found SDK DLLs at: {sdk_lib_path}")
    for dll in sdk_lib_path.glob('*.dll'):
        print(f"  Adding DLL: {dll.name}")
        infineon_binaries.append((str(dll), 'ifxradarsdk/lib'))
else:
    print(f"WARNING: SDK lib folder not found at {sdk_lib_path}")

# Add configs directory
configs_path = repo_root / 'configs'
if configs_path.exists():
    print(f"Found configs at: {configs_path}")
    configs_datas = [(str(configs_path), 'configs')]
else:
    print(f"WARNING: configs folder not found at {configs_path}")
    configs_datas = []

# Combine all data files
datas = infineon_datas + configs_datas

# Combine all binaries
binaries = infineon_binaries

print(f"\nTotal data files: {len(datas)}")
print(f"Total binaries: {len(binaries)}")

# Hidden imports
hiddenimports = [
    'ifxradarsdk',
    'ifxradarsdk.common',
    'ifxradarsdk.common.sdk_base',
    'ifxradarsdk.fmcw',
    'numpy',
    'scipy',
    'matplotlib',
    'PIL',
    'PyQt5',
]

a = Analysis(
    ['radar_gui_min.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PhyGO',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Console window for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PhyGO',
)