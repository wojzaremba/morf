function [recon_error, complexity] = approximate_layer(args)
    % args.layer_nt : Layer we are appromiximating
    % args.approx_type : Approximation function
    % args.approx_params : Any parameters required for approximated
    global plan
    layer = plan.layer{args.layer_nr};
    % Get approximation
    if (layer.on_gpu) 
        W = C_(CopyFromGPU, layer.gpu.vars.W);
    else
        W = plan.layer{args.layer_nr}.cpu.vars.W;
    end
    [Wapprox, recon_error] = monochromatic_approx(W, args.approx_params);
    
    % Replace weights in plan
    if (layer.on_gpu) 
        C_(CopyToGPU, layer.gpu.vars.W, Wapprox);
    else
        plan.layer{args.layer_nr}.cpu.vars.W = Wapprox;   
    end
    
    
end