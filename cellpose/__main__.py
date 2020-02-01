import numpy as np
import os
from cellpose import gui
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Suite2p parameters')
    parser.add_argument('--ops', default=[], type=str, help='options')
    parser.add_argument('--db', default=[], type=str, help='options')
    args = parser.parse_args()

    gui.run()
