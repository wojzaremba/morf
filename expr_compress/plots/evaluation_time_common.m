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
for i = 1:1
    plan.input.GetImage(0);
    ForwardPass(plan.input); 
    error = error + plan.classifier.GetScore(5);
    fprintf('%d / %d\n', error, i * plan.input.batch_size);
end

% matlab (single cpu)
% bs = 128
%     4.8529    19.1666    5.3316    4.3264    4.6245
% bs = 1
%     0.2990    0.2218    0.0648    0.0357    0.0406    


  
% cpp 
% bs = 128
%     9.0412    23.7738    9.9323    8.7355    8.4748
% bs = 1
%     0.9400    2.7637    1.3968    1.3709    1.4053
plan.time.fp([2, 5, 8, 10, 11])