# -*- mode: python ; coding: utf-8 -*-
import os
from pathlib import Path

block_cipher = None

# Find the Infineon SDK library directory
ifx_lib_path = Path.home() / 'phygo/venv/lib/python3.9/site-packages/ifxradarsdk/lib'

# Collect all .dylib files from the Infineon SDK
ifx_binaries = []
if ifx_lib_path.exists():
    for dylib in ifx_lib_path.glob('*.dylib'):
        # Add as (source, destination_folder)
        ifx_binaries.append((str(dylib), 'ifxradarsdk/lib'))

a = Analysis(
    ['radar_gui_min.py'],
    pathex=[],
    binaries=ifx_binaries,  # Include Infineon SDK libraries
    datas=[
        ('../configs', 'configs'),
        ('vex', 'vex'),
        ('processing_utils.py', '.'),
        ('InfineonManager.py', '.'),
    ],
    hiddenimports=[
        'PyQt5',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        'matplotlib',
        'matplotlib.backends.backend_qt5agg',
        'numpy',
        'scipy',
        'tensorflow',
        'tensorflow.keras',
        'websocket',
        'ifxradarsdk',
        'vex.aim',
        'vex.vex_types',
        'vex.vex_messages',
        'vex.vex_globals',
        'vex.settings',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'IPython',
        'jupyter',
        'notebook',
    ],
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
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
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

app = BUNDLE(
    coll,
    name='PhyGO.app',
    icon=None,
    bundle_identifier='com.phygo.radarml',
    info_plist={
        'CFBundleName': 'PhyGO',
        'CFBundleDisplayName': 'PhyGO Radar ML',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '10.13.0',
    },
)