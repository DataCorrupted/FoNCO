# FoNCO

First-order methods for constrained optimization(FoNCO)

## Dependencies

This work depends on CUTEst, please refer to [here](https://github.com/ralna/CUTEst/wiki) for installation details.

_scipy_ is required before this code can work. For Ubuntu users, please refer to the command below:

	sudo apt-get install python-pip
	pip install scipy

(Optional) To use result summary tools we provided, please install _panda_ too.

	pip install panda

(Optional) To see the generated summary, you need latex installed.

	sudo apt-get install texlive-full

This package is generally large(2G+), you can choose some sub-packages to install instead of all of them.

## Folders & Usage

### sif

This folder includes all the problems we tested. For detailed problems see the README in this folder.

Please copy all the sif file into this folder and run:

	./decode.py

It will call _sifdecoder_ and turn all your .sif file into a folder with decoded problems.

### cutest_py

A python-CUTEst interface written by Jianshan Wang. Some files are missing, please make sure that you have CUTEst installed and then run:

	./collect_files.sh

to collect all the files needed to compile the interface.

To compile run:

	./make_all ../sif

Currently it will only compile on 64bit linux machine. I will try to make it more generic in the future. 

It will compile the interface along with the .f file in each and every problem in the sif folder. It will dump the dynamic linked library in each problems folder generated in last step.

### src

All the codes should be here. Run:

	python run_experiment.py

to start an experiment. Human readable results and running logs will be put in _log/_, results for each problem will be put in _result/_.

(Optional) Use

	python output_summary.py

you can generate a latex file containing a table of all the results of the problems you just tested. A latex longtable of all the results will be generated in _summary/summary.tex_

(Optional) Use

	python get_time.py

A latex longtable of all problems time consumption and status will be generated in _summary/time.tex_

### summary

Just a folder to put a latex codes for generated summary.

## Results & External links

We have reached 113/126 accuracy on all HS problems in seconds. 

For more details you can refer to(The links are pointing to our Repos in Github.com for now, we will change this link once it is published):

[Our paper](https://github.com/DataCorrupted/slpls)

[Presentation](https://github.com/DataCorrupted/mopta2018)

