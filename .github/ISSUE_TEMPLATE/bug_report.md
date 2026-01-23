---
name: Bug report
about: Create a report to help us improve
title: "[BUG]"
labels: bug
assignees: ''

---
## Before you fill out this form:
**Did you review the FAQ?**
- [ ] Yes, I reviewed the FAQ on the [cellpose ReadTheDocs](https://cellpose.readthedocs.io/en/latest/faq.html) 

**Did you look through previous (open AND closed) issues posted on GH?**
- [ ] Yes, I [searched for related previous GH issues](https://github.com/MouseLand/cellpose/issues?q=is%3Aissue).

## Now fill this form out completely:
### **Describe the bug**
A clear and concise description of what the bug is.

### **To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error 

### **Run log**
Please post all command line/notebook output for us to understand the problem. For this please make sure you are running with verbose mode on. So command line, with `--verbose` tag, or in a notebook first run 
```from cellpose import io 
logger = io.logger_setup()
``` 
before running any Cellpose functions.

<details><summary>Run logs</summary>
your logs here.
</details>

### **System Information**
Please post the output of `python -m cellpose --version` or share the following details:
- OS and version (e.g., Windows 11, macOS 14.0, Ubuntu 22.04):
- Python version:
- Cellpose version:
- PyTorch version:
- GPU (if applicable):

### **Screenshots**
If applicable, add screenshots to help explain your problem.
