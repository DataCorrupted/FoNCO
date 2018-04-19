import sys

# Simple function used to debug. 
# input:
#		anything. It will print out anything you give.
def pause(*args, **kwargs):
	v = sys.version_info.major
	if v != 2 and v != 3:
		raise Exception("Unknown version of Py, am I in the future?")

	for item in args:
		print(item)
	for (key, item) in kwargs.items():
		print(key + ":\n", item)
	# Press Enter to continue.
	# The main code is using py2 while my util uses py3,
	# really frustring
	if v == 3:
		a = input()
	elif v == 2:
		a = raw_input()
