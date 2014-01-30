global plan
json = {};
json{1} = struct('batch_size', 3, 'rows', 8, 'cols', 10, 'depth', 2, 'type', 'TestInput');
json{2} = struct('local_2d_patch', struct('patch_rows', 4, 'patch_cols', 4), 'function', 'RELU', 'depth', 8, 'num_image_colors', 4, 'type', 'MonoConv');
Plan(json, [], 0, 'single');
plan.layer{2}.cpu.vars.perm = randperm(json{2}.depth)' - 1;

plan.input.GetImage(1);
plan.layer{2}.cpu.vars.X = single(plan.layer{1}.cpu.vars.out);
plan.layer{2}.cpu.vars.Cmono = single(randn(size(plan.layer{2}.cpu.vars.Cmono)));
plan.layer{2}.cpu.vars.Wmono = single(randn(size(plan.layer{2}.cpu.vars.Wmono)));
plan.layer{2}.FP();
outFP = plan.layer{2}.cpu.vars.out;
plan.layer{2}.cpu.vars.out(:) = 0;

plan.layer{2}.FPcpp();
outFPcpp = plan.layer{2}.cpu.vars.out;

squeeze(sum(sum(sum(outFP, 1), 2), 3))

assert(norm(outFP(:) - outFPcpp(:)) < 1e-4);

