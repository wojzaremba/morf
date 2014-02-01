clc
clear all
global plan debug
debug = 2;
json = ParseJSON('plans/imagenet_matthew.txt');
        
json{1}.batch_size = 128;
Plan(json, '~/imagenet_data/imagenet_matthew', 0, 'single');
plan.only_fp = 1;
plan.training = 0;
plan.input.step = 1;
error = 0;
for i = 1:8
    plan.input.GetImage(0);
    ForwardPass(plan.input); 
    error = error + plan.classifier.GetScore(5);
    fprintf('%d / %d\n', error, i * plan.input.batch_size);
end



% error should be 20 / 128.
% 1. More data to compute mean.
% 2. Regenerate stuff with crop from the central (maybe it is resized).