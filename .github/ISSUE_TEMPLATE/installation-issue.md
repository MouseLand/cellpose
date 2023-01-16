---
name: Installation issue
about: problems installing cellpose with or without GPU support
title: "[INSTALL]"
labels: install
assignees: ''

---

**Install problem**
Let us know what issues you are having with installation.

**Environment info**
Please run `conda list` in your cellpose environment in the terminal / anaconda prompt to let us know your software versions.

**Run log**
Please post all command line/notebook output for us to understand the problem. For this please make sure you are running with verbose mode on. So command line, with `--verbose` tag, or in a notebook first run 

```from cellpose import io 
logger = io.logger_setup()
```

before running any Cellpose functions.
