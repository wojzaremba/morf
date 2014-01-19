function GPUVerification()
clc;
clear all
plan_json = ParseJSON('plans/tests.txt');
for i = 1 : length(plan_json)
    jsons = {};
    jsons{2} = plan_json{i};
    if (strcmp(plan_json{i}.type, 'Softmax'))
        continue; % XXX : Not fully working.
        jsons{1} = struct('depth', 20, 'rows', 1, 'cols', 1, 'batch_size', 6, 'type', 'TestInput');        
    elseif (strcmp(plan_json{i}.type, 'Conv'))
        jsons{1} = struct('batch_size', 128, 'rows', 16, 'cols', 16, 'depth', 32, 'type', 'TestInput');
        jsons{2} = struct('type', 'Conv', 'depth', 64, 'function', 'RELU', 'local_2d_patch', struct('patch_rows', 8, 'patch_cols', 8, 'stride_rows', 1, 'stride_cols', 1, 'padding_rows', 0, 'padding_cols', 0));        
    else
        jsons{1} = struct('depth', 32, 'rows', 16, 'cols', 16, 'batch_size', 6, 'type', 'TestInput');
    end       
    Plan(jsons, [], 1);
    VerifyLayerFP();
    VerifyLayerBP();
end
end

function VerifyLayerFP()
global plan;
layer = plan.layer{end};
if (~ismethod(layer, 'FP_'))
    return;
end
fprintf('Verifing consistency of FP for %s layer\n\n', layer.type);
plan.input.GetImage(1);
layer.cpu.vars.X = plan.input.cpu.vars.out;
layer.gpu.vars.X = plan.input.gpu.vars.out;
layer.FP();
out = layer.cpu.vars.out;
layer.FP_();
out_ = C_(CopyFromGPU, layer.gpu.vars.out);
assert(norm(out(:) - out_(:)) / max(norm(out(:)), 1) < 1e-4);
end


function VerifyLayerBP()
global plan;
layer = plan.layer{end};
if (~ismethod(layer, 'BP_'))
    return;
end
fprintf('Verifing consistency of BP for %s layer\n\n', layer.type);
plan.input.GetImage(1);
layer.cpu.vars.X = plan.input.cpu.vars.out;
layer.gpu.vars.X = plan.input.gpu.vars.out;
node_data = single(randn([plan.input.batch_size, layer.dims()]));
layer.cpu.dvars.out = node_data;
layer.BP();
dvars = layer.cpu.dvars;
C_(CopyToGPU, plan.gid, node_data);
layer.gpu.dvars.out = plan.gid;
layer.BP_();

f = fields(dvars);
for i = 1:length(f)
	d = eval(sprintf('dvars.%s;', f{i}));
    gid = eval(sprintf('layer.gpu.dvars.%s;', f{i}));
	d_ = C_(CopyFromGPU, gid);
	assert(norm(d(:) - d_(:)) < 1e-4); 
end
end
