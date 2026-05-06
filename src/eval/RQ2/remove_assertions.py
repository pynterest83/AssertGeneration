import os
import sys


def remove_asserts(path):
    for root, _, files in os.walk(path):
        for name in files:
            if name.endswith("_ESTest.java"):
                filepath = os.path.join(root, name)
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                with open(filepath, 'w', encoding='utf-8') as f:
                    for line in lines:
                        if line.strip().startswith("assert"):
                            line = '//' + line
                        f.write(line)


if __name__ == "__main__":
    remove_asserts(sys.argv[1])
