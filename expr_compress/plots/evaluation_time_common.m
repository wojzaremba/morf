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
%     0.0012    4.8529    2.6598    4.0902   19.1666    1.4867    2.1572    5.3316    0.7628    4.3264    4.6245    0.1798    0.8683    0.1842    0.0524    0.0148

  
% cpp
%     0.0012    9.1418    2.9977    4.2022   24.9191    1.3220    2.1612    9.9182    0.6945    8.8772    8.8612    0.1631    0.7101    0.1401    0.0361    0.0095
