# -*- mode: python ; coding: utf-8 -*-

import os
import os.path

import PyInstaller.compat
import PyInstaller.utils.hooks

block_cipher = None

hiddenimports = []

hiddenimports += PyInstaller.utils.hooks.collect_submodules("scipy")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("scipy.ndimage")

a = Analysis(['cli.py'],
             pathex=['C:\\Users\\carse\\github\\cellpose'],
             binaries=[],
             datas=[('C:/Users/carse/anaconda3/Lib/site-packages/mxnet/*', './mxnet')],
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

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='cellpose',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True,
          icon='cellpose/logo/cellpose.ico')