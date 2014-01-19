clear all;
addpath(genpath('.'));
global plan
json = ParseJSON('plans/imagenet.txt');
json{1}.batch_size = 128;
Plan(json, 'trained/imagenet');
plan.training = 0;
plan.input.step = 1;
plan.input.GetImage(0);
ForwardPass(plan.input);
fprintf('errors = %d / %d\n', plan.classifier.GetScore(5), json{1}.batch_size);
