global plan
addpath(genpath('.'));
json = ParseJSON('plans/mnist_simple.txt');
json{1}.batch_size = 100;
Plan(json, [], 0);
Run();



% XXX: specify -arch=sm_35 -O3


% Tasks
% 3. Create branched objective.
% 4. Write small end to end tests.
