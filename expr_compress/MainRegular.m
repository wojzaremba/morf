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


% ************************************************************************************************************************incorrect = 7805, all = 12000
% ************************************************************************************************************************incorrect = 11599, all = 24000
% ************************************************************************************************************************incorrect = 13443, all = 36000
% ************************************************************************************************************************incorrect = 14784, all = 48000
% ************************************************************************************************************************incorrect = 16008, all = 60000
% 
% Epoch took = 36.232734
% Testing:
% ****************************************************************************************************
% epoch = 1, incr_test = 944, err = 0.094400