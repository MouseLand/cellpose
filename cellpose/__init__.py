from cellpose.version import version, version_str
import logging

# set base `cellpose` logger
logging.getLogger(__name__).addHandler(logging.NullHandler())