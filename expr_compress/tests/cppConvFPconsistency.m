clc;
clear all;
global plan
json = {};
json{1} = struct('batch_size', 128, 'rows', 15, 'cols', 15, 'depth', 96, 'type', 'TestInput');
json{2} = struct('local_2d_patch', struct('patch_rows', 4, 'patch_cols', 4, 'stride_rows', 2, 'stride_cols', 2, 'padding_rows', 1, 'padding_cols', 1), 'function', 'RELU', 'depth', 5, 'type', 'Conv');
Plan(json, [], 0, 'single');

plan.input.GetImage(1);
plan.layer{2}.cpu.vars.X = single(plan.layer{1}.cpu.vars.out);
plan.layer{2}.FP_old();
outFP = plan.layer{2}.cpu.vars.out;
plan.layer{2}.cpu.vars.out(:) = 0;

plan.layer{2}.FP();
outFPcpp = plan.layer{2}.cpu.vars.out;

assert(norm(outFP(:) - outFPcpp(:)) < 1e-4);