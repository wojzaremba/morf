classdef Plan < handle
    properties
        jsons
        debug
        lr
        momentum
        training
        tiny
        stats
        layer
        input
        costs
        classifier
        gid
        time
        default_on_gpu
        maxIter
        upload_weights
        all_uploaded_weights
        verbose
        only_fp
        incr_tests 
        incr_trains
    end
    
    methods
        function obj = Plan(param1, weights, default_on_gpu)
            if (ischar(param1))
                jsons = ParseJSON(param1);
            else
                jsons = param1;
            end
            if (exist('default_on_gpu', 'var'))
                obj.default_on_gpu = default_on_gpu;
            else
                obj.default_on_gpu = 0;
            end
            obj.jsons = jsons;
            obj.gid = 0;  
            obj.only_fp = 0;
            try
                C_(CleanGPU);
            catch
                fprintf('GPU not available\n');              
            end
            obj.debug = 0;
            obj.lr = 0;
            obj.training = 0;
            obj.tiny = exp(-100);
            randn('seed', 1);
            rand('seed', 1);            
            obj.layer = {};
            if (exist('weights', 'var')) && (~isempty(weights))
                obj.all_uploaded_weights = load(weights);
            end            
            global plan cuda
            plan = obj;     
            cuda = zeros(2, 1);            
            obj.stats = struct('total_vars', 0, 'total_learnable_vars', 0, 'total_vars_gpu', 0);
            for i = 1:length(jsons)
                json = jsons{i};
                if (strcmp(json.type(), 'Spec'))
                    obj.lr = json.lr;
                    obj.momentum = json.momentum;
                else
                    obj.layer{end + 1} = eval(sprintf('%s(json);', json.type()));
                end
            end
            fprintf('Total number of\n\ttotal learnable vars = %d\n\ttotal vars = %d\n\ttotal vars on the gpu = %d\n', obj.stats.total_learnable_vars, obj.stats.total_vars, obj.stats.total_vars_gpu);            
            obj.all_uploaded_weights = [];
        end        
        
        function gid = GetGID(obj)
            gid = obj.gid; 
            obj.gid = obj.gid + 1;            
        end
        
    end
end