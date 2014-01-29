clear all
global plan
jsons{1} = struct('batch_size', 2, 'rows', 1, 'cols', 1, 'depth', 4, 'type', 'TestInput');
jsons{2} = struct('type', 'FC', 'depth', 5, 'function', 'RELU');
jsons{2}.on_gpu = 0;
Plan(jsons, [], 1);
plan.lr = 11;
plan.momentum = 0.8;

layer = plan.layer{end};
plan.input.GetImage(1);
layer.cpu.vars.X = plan.input.cpu.vars.out;
layer.FP();

back_in = single(randn([size(layer.cpu.vars.X, 1), layer.dims]));
layer.cpu.dvars.out = back_in;

layer.BP();
W_cpu_init = layer.cpu.vars.W;
layer.Update();
W_cpu = layer.cpu.vars.W;
W_cpu_accum = layer.cpu.accum.W;
W_cpu_d = layer.cpu.dvars.W;


jsons{2}.on_gpu = 1;
Plan(jsons, [], 1);
plan.lr = 11;
plan.momentum = 0.8;

layer = plan.layer{end};
plan.input.GetImage(1);
layer.cpu.vars.X = plan.input.cpu.vars.out;
layer.FP_();

layer.gpu.dvars.out = plan.GetGID();
C_(CopyToGPU, layer.gpu.dvars.out, back_in);

layer.BP_();
W_gpu_init = C_(CopyFromGPU, layer.gpu.vars.W);
layer.Update();
W_gpu = C_(CopyFromGPU, layer.gpu.vars.W);
W_gpu_accum = C_(CopyFromGPU, layer.gpu.accum.W);
W_gpu_d = C_(CopyFromGPU, layer.gpu.dvars.W);

assert(norm(W_cpu_d(:) - W_gpu_d(:)) < 1e-4);
assert(norm(W_cpu_accum(:) - W_gpu_accum(:)) < 1e-4);
assert(norm(W_cpu_init(:) - W_gpu_init(:)) < 1e-4);
assert(norm(W_cpu(:) - W_gpu(:)) < 1e-4);
assert(norm(W_cpu(:) - W_cpu_init(:)) > 1e-4);