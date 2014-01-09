function ForwardPass(obj)
global plan

if (obj.on_gpu)
    if (obj.layer_nr > 1) 
        if (plan.layer{obj.layer_nr - 1}.on_gpu == 0)
            C_(CopyToGPU, obj.gpu.vars.X,  single(plan.layer{obj.layer_nr - 1}.cpu.vars.out));
        else
            obj.gpu.vars.X = plan.layer{obj.layer_nr - 1}.gpu.vars.out;            
        end
    end
    C_(StartTimer);
    obj.FP_();
    lapse = C_(StopTimer);
else
    if (obj.layer_nr > 1) 
        if (plan.layer{obj.layer_nr - 1}.on_gpu == 1)
            obj.cpu.vars.X = reshape(C_(CopyFromGPU, plan.layer{obj.layer_nr - 1}.gpu.vars.out), size(obj.cpu.vars.X));
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
    ForwardPass(obj.next{k});   
end
end