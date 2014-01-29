global plan
addpath(genpath('.'));
json = ParseJSON('plans/mnist_simple.txt');
json{1}.batch_size = 100;
json{3}.p = 0;
json{5}.p = 0;
Plan(json, [], 0);
RunRegular();

% 1. Reproduce results of Hinton, and Wan Li.
% 2. Check various ways of droping subspaces / eigenvalues, removing
% eigenvalues. In case of droping do it on much finner scale then epoch
% (write a layer).



% 180 errors without Dropout
% two dropout gives error : 165


% XXX : Look into spectral properties of dropout and non-dropout matrices !!!!
% XXX : Look into vectors corresponding to the major eigenvalues !!


% Put it all to understanding !!!!!!!