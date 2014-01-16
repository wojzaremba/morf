classdef NoApproximation < Approximation
    properties        
    end
    
    methods
        function obj = NoApproximation(suffix, approx_vars, cuda_vars)
            obj@Approximation(suffix, approx_vars, cuda_vars);
            obj.name = 'no_approximation';
        end
        
        function ret = VerifyCombination(obj, approx_vars, cuda_vars)
            ret = true;
        end        
        
        function [Wapprox, ret] = Approx(obj, params)            
            global plan;
            Wapprox = plan.layer{2}.cpu.vars.W;          
            ret.Wapprox = Wapprox;
            ret.layer_nr = 2;
            ret.vars.W = Wapprox;
        end
        
        function [test_error, time] = RunModifConv(obj, args)        
            global plan
            layer_orig = plan.layer{args.layer_nr};
            json = layer_orig.json;
            json.on_gpu = Val(args, 'on_gpu', 1);
            layers = {};
            layers = plan.layer;
            plan.layer = {};
            for i = 1 : length(layers)
                if (i ~= args.layer_nr)
                    plan.layer{i} = layers{i};
                else
                    plan.layer{args.layer_nr} = eval(sprintf('Conv(json)'));                                            
                end
            end            
            plan.layer{args.layer_nr - 1}.next = {plan.layer{args.layer_nr}};            
            plan.layer{args.layer_nr}.next = {plan.layer{args.layer_nr + 1}};
            obj.SetLayerVars(args);     
            plan.input.GetImage(0);
            ForwardPass(plan.input);
            test_error = plan.classifier.GetScore();            
            time = plan.time.fp(2);
            plan.layer = layers;
            plan.layer{args.layer_nr - 1}.next = {plan.layer{args.layer_nr}};            
            plan.layer{args.layer_nr}.next = {plan.layer{args.layer_nr + 1}};
            C_(CleanGPU);
        end
    end
    
end