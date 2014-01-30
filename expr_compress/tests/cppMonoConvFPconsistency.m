global plan
json = {};
for bs = [1, 12]
    json{1} = struct('batch_size', bs, 'rows', 8, 'cols', 10, 'depth', 2, 'type', 'TestInput');
    json{2} = struct('local_2d_patch', struct('patch_rows', 4, 'patch_cols', 4), 'function', 'RELU', 'depth', 8, 'num_image_colors', 4, 'type', 'MonoConv');

    Plan(json, [], 0, 'single');
    seed = RandStream('mt19937ar','Seed',0);

    plan.input.GetImage(1);
    plan.layer{2}.cpu.vars.X = single(plan.layer{1}.cpu.vars.out);
    plan.layer{2}.cpu.vars.Cmono = single(randn(size(plan.layer{2}.cpu.vars.Cmono)));
    plan.layer{2}.cpu.vars.Wmono = single(randn(size(plan.layer{2}.cpu.vars.Wmono)));
    plan.layer{2}.cpu.vars.perm = single(randperm(seed, json{2}.depth)' - 1);
    plan.layer{2}.cpu.vars.B = single(randn(size(plan.layer{2}.cpu.vars.B)));
    plan.layer{2}.FP();
    outFP = plan.layer{2}.cpu.vars.out;
    plan.layer{2}.cpu.vars.out(:) = 0;

    plan.layer{2}.FPcpp();
    outFPcpp = plan.layer{2}.cpu.vars.out;

    prev_dim = plan.layer{2}.prev_dim();
    v = plan.layer{2}.cpu.vars;
    X = v.X;  
    bs = size(X, 1);
    % Color transform
    X = reshape(X, [bs * prev_dim(1) * prev_dim(2), prev_dim(3)]) * v.Cmono;

    assert(norm(X(:) - v.Xmono(:)) < 1e-4);

    assert(norm(outFP(:) - outFPcpp(:)) < 1e-4);
end