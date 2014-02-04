function ForwardPass(obj)
global plan
printf(2, 'FP for %s\n', obj.name);
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
    % XXXXXXX
%     if (strcmp(obj.type, 'Conv'))
%         tmp = obj.cpu.vars.out;
%         obj.cpu.vars.out(:) = 0;
%         obj.FPcpp();
%         norm(obj.cpu.vars.out(:) - tmp(:)) / norm(obj.cpu.vars.out(:))
%     end
    lapse = toc(fptic);
end
plan.time.fp(max(plan.input.step - 1, 1), obj.layer_nr) = lapse;
for k = 1:length(obj.next)
    ForwardPass(obj.next{k});   
end
end
