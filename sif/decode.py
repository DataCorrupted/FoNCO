#!/usr/bin/python

import os
import sys
import shutil
def transform(path):
	if not os.path.exists(path):
		print("Error, invalid path.")
	files = os.listdir(path)
	for f in files:
		if f[-4:] != '.SIF':
			continue;
		prob = f[:-4]
		if not os.path.exists(prob):
			os.mkdir(prob)
		shutil.move(f, prob+"/"+f)
		os.chdir(prob)
		os.system("sifdecoder " + f)
		os.chdir("..")
		
def main(args, argv):
	if args == 1:
		print("Warning, no path given. Processing every file in current folder.");
		path = "."
	else:
		path = argv[1];
	transform(path)

if __name__ == "__main__":
	main(len(sys.argv), sys.argv)