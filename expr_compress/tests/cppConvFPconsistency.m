global plan
json = {};
json{1} = struct('batch_size', 128, 'rows', 8, 'cols', 10, 'depth', 4, 'type', 'TestInput');
json{2} = struct('local_2d_patch', struct('patch_rows', 4, 'patch_cols', 4), 'function', 'RELU', 'depth', 5, 'type', 'Conv');
Plan(json, [], 0, 'single');


plan.input.GetImage(1);
plan.layer{2}.cpu.vars.X = single(plan.layer{1}.cpu.vars.out);
plan.layer{2}.FP();
outFP = plan.layer{2}.cpu.vars.out;
plan.layer{2}.cpu.vars.out(:) = 0;

plan.layer{2}.FPcpp();
outFPcpp = plan.layer{2}.cpu.vars.out;

assert(norm(outFP(:) - outFPcpp(:)) < 1e-4);