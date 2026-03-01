import pandas as pd
import os, re, csv, argparse, sys
from pathlib import Path
from collections import defaultdict


def remove_asserts(path):
    for path, subdirs, files in os.walk(path):
        for name in files:

            if name.endswith("_ESTest.java"):
                test_class = os.path.join(path, name)
                print(test_class)
                with open(test_class, 'r+') as f:
                    lines = f.readlines()
                    f.seek(0)
                    f.truncate()
                    for line in lines:
                        if line.strip().startswith("assert"):
                            line = "//"+line
                        f.write(line)
                f.close()



if __name__ == "__main__":
    remove_asserts(sys.argv[1])