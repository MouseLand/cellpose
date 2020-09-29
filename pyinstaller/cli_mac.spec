# -*- mode: python ; coding: utf-8 -*-
# make clean environment + pip install -e .[gui]

import os, sys
import os.path

import PyInstaller.compat
import PyInstaller.utils.hooks

block_cipher = None

hiddenimports = []

hiddenimports += PyInstaller.utils.hooks.collect_submodules("scipy")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("scipy.ndimage")

a = Analysis(['cli.py'],
             pathex=['/Users/loaner/github/cellpose'],
             binaries=[],
             datas=[('/opt/anaconda3/envs/cellpose/lib/python3.7/site-packages/mxnet/*', 
                        './mxnet')],
             hiddenimports=hiddenimports,
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

if sys.platform == 'darwin':
    exe = EXE(pyz,
            a.scripts,
            a.binaries,
            a.zipfiles,
            a.datas,
            name='cellpose',
            debug=False,
            strip=False,
            upx=True,
            runtime_tmpdir=None,
            console=True,
            icon='cellpose/logo/cellpose.icns')

# Package the executable file into .app if on OS X
if sys.platform == 'darwin':
    app = BUNDLE(exe,
                name='cellpose.app',
                info_plist={
                  'NSHighResolutionCapable': 'True'
                },
                icon='cellpose/logo/cellpose.icns')