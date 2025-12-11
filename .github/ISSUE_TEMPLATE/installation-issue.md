---
name: Installation issue
about: problems installing cellpose with or without GPU support
title: "[INSTALL]"
labels: install
assignees: ''

---
## Before you fill out this form:
**Did you review the FAQ?**
- [ ] Yes, I reviewed the FAQ on the [cellpose ReadTheDocs](https://cellpose.readthedocs.io/en/latest/faq.html) 

**Did you look through previous (open AND closed) issues posted on GH?**
- [ ] Yes, I [searched for related previous GH issues](https://github.com/MouseLand/cellpose/issues?q=is%3Aissue).

## Now fill this form out completely:
### **Install problem**
Let us know what issues you are having with installation.

### **Environment info**
Please run `conda list` in your cellpose environment in the terminal / anaconda prompt to let us know your software versions.

<details><summary>Conda list output</summary>
your conda list output
</details>

### **Run log**
Please post all command line/notebook output for us to understand the problem. For this please make sure you are running with verbose mode on. So command line, with `--verbose` tag, or in a notebook first run 

```from cellpose import io 
logger = io.logger_setup()
```

before running any Cellpose functions.
