global plan
json = {};
json{1} = struct('batch_size', 3, 'rows', 8, 'cols', 10, 'depth', 4, 'type', 'TestInput');
json{2} = struct('local_2d_patch', struct('patch_rows', 4, 'patch_cols', 4), 'function', 'RELU', 'depth', 5, 'type', 'Conv');
% json{1} = struct('batch_size', 1, 'rows', 2, 'cols', 2, 'depth', 2, 'type', 'TestInput');
% json{2} = struct('local_2d_patch', struct('patch_rows', 2, 'patch_cols', 2), 'function', 'RELU', 'depth', 1, 'type', 'Conv');
Plan(json, [], 0, 'single');


plan.input.GetImage(1);
plan.layer{2}.cpu.vars.X = single(plan.layer{1}.cpu.vars.out);
plan.layer{2}.FP();
outFP = plan.layer{2}.cpu.vars.out;
plan.layer{2}.cpu.vars.out(:) = 0;

plan.layer{2}.FPcpp();
outFPcpp = plan.layer{2}.cpu.vars.out;

assert(norm(plan.layer{2}.cpu.vars.forward_act(:) - outFPcpp(:)) < 1e-4);