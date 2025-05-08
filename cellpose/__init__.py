from cellpose.version import version, version_str
message = f'''\n
Welcome to CellposeSAM, cellpose v{version_str}! The neural network component of
CPSAM is much larger than in previous versions and CPU excution is slow. 
We encourage users to use GPU/MPS if available. \n\n''' 
print(message)
