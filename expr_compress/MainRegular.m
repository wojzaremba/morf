global plan incr_tests incr_trains
incr_tests = [];
incr_trains = [];
addpath(genpath('.'));
json = ParseJSON('plans/mnist_simple.txt');
json{1}.batch_size = 100;
Plan(json, [], 0);
RunRegular();