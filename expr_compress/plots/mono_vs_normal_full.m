clc
clear all
reps = 1;
global plan debug
debug = 2;
json = ParseJSON('plans/imagenet_matthew.txt');
json{1}.batch_size = 128;
Plan(json, '~/imagenet_data/imagenet_matthew', 0, 'single');
W = plan.layer{2}.cpu.vars.W;
B = plan.layer{2}.cpu.vars.B;
plan.only_fp = 1;
plan.training = 0;
plan.input.step = 1;
error = 0;
for i = 1 : reps
    plan.input.GetImage(0);
    ForwardPass(plan.input); 
    error = error + plan.classifier.GetScore(5);
    fprintf('%d / %d\n', error, i * plan.input.batch_size);
end

plan.time.fp

json = ParseJSON('plans/imagenet_matthew.txt');
json{2}.type = 'MonoConv';
json{2}.num_image_colors = 24;        
json{1}.batch_size = 128;
Plan(json, '~/imagenet_data/imagenet_matthew', 0, 'single');

fprintf('Computing approximation\n');
[Wapprox, Wmono_, Cmono, perm] = monochromatic_approx(double(W), struct('num_colors', json{2}.num_image_colors, 'even', 1, 'start', 'sample'));
fprintf('Approximated\n');

perm = perm' - 1;
Wmono = permute(Wmono_, [2, 3, 1]);

plan.layer{2}.cpu.vars.Cmono = single(Cmono);
plan.layer{2}.cpu.vars.Wmono = single(Wmono);
plan.layer{2}.cpu.vars.perm = single(perm);
plan.layer{2}.cpu.vars.B = single(B);

plan.only_fp = 1;
plan.training = 0;
plan.input.step = 1;
error = 0;
for i = 1 : reps
    plan.input.GetImage(0);
    ForwardPass(plan.input); 
    error = error + plan.classifier.GetScore(5);
    fprintf('%d / %d\n', error, i * plan.input.batch_size);
end

plan.time.fp

