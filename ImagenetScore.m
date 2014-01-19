% clear all;
global plan debug
addpath(genpath('.'));
Plan('plans/imagenet.txt', 'trained/imagenet');
plan.training = 0;
plan.only_fp = 1;
debug = 2;

error = 0;
plan.input.step = 1;
for i = 1:8
    plan.input.GetImage(0);
    ForwardPass(plan.input); 
    error = error + plan.classifier.GetScore(5);
    fprintf('%d / %d\n', error, i * plan.input.batch_size);
end
