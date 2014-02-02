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

plan.time.fp

% matlab
%       0.0011    3.9622    2.2095    1.7071   13.3983    1.1448    0.9304    2.8241    0.5330    1.9286    2.5859    0.1397    0.5469    0.1231    0.0398    0.0126

% cpp, MKL
%       0.0011    3.0658    2.3650    1.7210    3.4963    1.1215    0.7761    1.5675    0.5207    1.2880    1.2266    0.1154    0.3881    0.0702    0.0171    0.0094

% 0. Compile with MKL.
% 1. Przepisz to tak, zeby bylo 1-1 z kodem w matlab i zacznij modyfikacje
% od tego momentu