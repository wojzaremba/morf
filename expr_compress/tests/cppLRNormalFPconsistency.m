clc;
clear all;
global plan
json = {};
json{1} = struct('batch_size', 128, 'rows', 4, 'cols', 4, 'depth', 96, 'type', 'TestInput');
json{2} = struct('k', 2, 'n', 5, 'alpha', 2e-4, 'beta', 0.75, 'one2one', true, 'type', 'LRNormal');

Plan(json, [], 0, 'single');

plan.input.GetImage(1);
plan.layer{2}.cpu.vars.X = single(plan.layer{1}.cpu.vars.out);
plan.layer{2}.FPmatlab();
outFP = plan.layer{2}.cpu.vars.out;
plan.layer{2}.cpu.vars.out(:) = 0;

plan.layer{2}.FP();
outFPcpp = plan.layer{2}.cpu.vars.out;

assert(norm(outFP(:) - outFPcpp(:)) / norm(outFP(:)) < 1e-4);