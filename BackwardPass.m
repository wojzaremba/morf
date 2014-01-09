function BackwardPass(obj)
global plan
if (obj.on_gpu)
    C_(StartTimer);
    obj.BP_();
    lapse = C_(StopTimer);
else
    fptic = tic;
    obj.BP();
    lapse = toc(fptic);
end
plan.time.bp(obj.layer_nr) = lapse;
obj.Update();
if (obj.layer_nr <= 2)
    return;
end
for k = 1:length(obj.prev)    
    if (obj.on_gpu) && (obj.prev{k}.on_gpu)
        obj.prev{k}.gpu.dvars.out = obj.gpu.dvars.X;
    elseif (obj.on_gpu) && (~obj.prev{k}.on_gpu)
        obj.prev{k}.cpu.dvars.out = C_(CopyFromGPU, obj.gpu.dvars.X);
    elseif (~obj.on_gpu) && (~obj.prev{k}.on_gpu)
        obj.prev{k}.cpu.dvars.out = obj.cpu.dvars.X;
    elseif (~obj.on_gpu) && (obj.prev{k}.on_gpu)
        C_(CopyToGPU, obj.prev{k}.gpu.dvars.out, obj.cpu.dvars.X);
    end 
    BackwardPass(obj.prev{k});
end
end
