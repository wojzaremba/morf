function ForwardPassApprox(obj)
global plan

% XXX : This won't work if we want to do some of the original "non-slim"
% layers on the GPU (eg. regular conv, pooling, etc.).

if (obj.on_gpu)
    if (obj.layer_nr > 1) 
        if (plan.layer{obj.layer_nr - 1}.on_gpu == 0)
            Capprox_gen(CopyToGPU, obj.gpu.vars.X,  single(plan.layer{obj.layer_nr - 1}.cpu.vars.out));
        else
            obj.gpu.vars.X = plan.layer{obj.layer_nr - 1}.gpu.vars.out;            
        end
    end
    Capprox_gen(StartTimer);
    obj.FP_();
    lapse = Capprox_gen(StopTimer);
else
    if (obj.layer_nr > 1) 
        if (plan.layer{obj.layer_nr - 1}.on_gpu == 1)
            obj.cpu.vars.X = reshape(Capprox_gen(CopyFromGPU, plan.layer{obj.layer_nr - 1}.gpu.vars.out), size(obj.cpu.vars.X));
        else
            obj.cpu.vars.X = plan.layer{obj.layer_nr - 1}.cpu.vars.out;              
        end
    end   
    fptic = tic;
    obj.FP();
    lapse = toc(fptic);
end
plan.time.fp(obj.layer_nr) = lapse;
for k = 1:length(obj.next)
    ForwardPassApprox(obj.next{k});   
end
end
