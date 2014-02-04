clc;
clear all;
global plan
json = {};
json{1} = struct('batch_size', 128, 'rows', 8, 'cols', 8, 'depth', 3, 'type', 'TestInput');
json{2} = struct('local_2d_patch', struct('patch_rows', 4, 'patch_cols', 4, 'stride_rows', 1, 'stride_cols', 1), 'type', 'MaxPooling');
Plan(json, [], 0, 'single');

plan.input.GetImage(1);
plan.layer{2}.cpu.vars.X = single(plan.layer{1}.cpu.vars.out);
plan.layer{2}.FPmatlab();
outFP = plan.layer{2}.cpu.vars.out;
plan.layer{2}.cpu.vars.out(:) = 0;

plan.layer{2}.FP();
outFPcpp = plan.layer{2}.cpu.vars.out;

assert(norm(outFP(:) - outFPcpp(:)) / norm(outFP(:)) < 1e-4);