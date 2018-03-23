#!/usr/bin/python
import os
import shutil
import sys
from subprocess import call


def main(sifDir):
    alldir = os.listdir(sifDir)
    alldir = [z for z in alldir if os.path.isdir(os.path.join(sifDir, z))]
    for z in alldir:
            call(['make', 'OBJDIR='+os.path.join(sifDir, z)])


if __name__ == '__main__':
    sifDir = os.path.abspath(sys.argv[1])

    main(sifDir)

