morf
====

Download imagenet weights from http://www.cs.nyu.edu/~zaremba/imagenet.mat and placed them in directory trained.

Build cuda wrappers by running Makefile in cuda directory.

Verify your setup by executing tests : RunTests.m

Score 128 imagenet test images with (expected result 108 correctly predicted out of 128) : ImagenetScore.m

For now on training doesn't work (no point to run Main.m)

