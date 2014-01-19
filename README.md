morf
====

Download all heavy files from http://www.cs.nyu.edu/~zaremba/morf_rest and placed them in the main directory. This are files with weights, and test images.

Build cuda wrappers by running Makefile in cuda directory.

Verify your setup by executing tests : RunTests.m

Score 128 imagenet test images with (expected result 108 correctly predicted out of 128) : ImagenetScore.m


