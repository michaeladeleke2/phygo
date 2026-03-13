# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

mpl_datas, mpl_binaries, mpl_hiddenimports = collect_all('matplotlib')
numpy_datas, numpy_binaries, numpy_hiddenimports = collect_all('numpy')

a = Analysis(
    ['radar_gui_min.py'],
    pathex=[],
    binaries=[] + mpl_binaries + numpy_binaries,
    datas=[
        ('../configs', 'configs'),
        ('vex', 'vex'),
        ('processing_utils.py', '.'),
        ('InfineonManager.py', '.'),
    ] + mpl_datas + numpy_datas,
    hiddenimports=[
        'numpy', 'scipy', 'matplotlib', 'matplotlib.pyplot',
        'matplotlib.backends.backend_qt5agg',
        'matplotlib.backends.backend_agg',
        'PyQt5', 'PyQt5.QtCore', 'PyQt5.QtGui', 'PyQt5.QtWidgets', 'PyQt5.sip',
        'tensorflow', 'tensorflow.keras',
        'torch', 'torch.nn', 'torch.nn.functional',
        'torchvision', 'transformers',
        'websocket', 'websocket._core',
        'ifxradarsdk',
        'vex.aim', 'vex.vex_types', 'vex.vex_messages', 'vex.vex_globals', 'vex.settings',
        'PIL', 'PIL.Image', 'pandas', 'sklearn', 'pkg_resources',
    ] + mpl_hiddenimports + numpy_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter', 'IPython', 'jupyter', 'notebook',
        'optree',
        'mne', 'brainflow',
        'sympy', 'mpmath',
        'torchgen', 'functorch',
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
