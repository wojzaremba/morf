global plan
addpath(genpath('.'));
json = ParseJSON('plans/mnist_x3.txt');
json{1}.batch_size = 100;
Plan(json, [], 0);
Run();


% Result with 200(LINEAR)-10 : 741
% Result with 200(X3)-10 : 360
% Result with 200(RELU)-10 : 201
% Result with 600(X3)-10 : 312


% Tasks
% 3. Create branched objective.
% 4. Write small end to end tests.

