% XXX : missing serialization.

classdef ApproxLog < handle
    properties
        name % Name of approximation for which this log is defined.
        cudas_logs % hashmap of CudasConfig
    end
    
    methods
        function obj = ApproxLog(name)
            obj.name = name;
            obj.cudas_logs = containers.Map('KeyType', 'char', 'ValueType', 'any');
        end
        
        function obj = Add(obj, cudas_config)
            assert(0);
        end                
        
        function ret = IsProcessed(obj, approx_vars, cuda_vars)
            func_str = struct2str(approx_vars);
            if (~obj.cudas_logs.isKey(func_str))
                ret = false;
                return;
            end
            assert(0);
%             cudas_config = 
        end
        
        function SaveApproxInfo(obj, approx_vars, info)
            str = struct2str(approx_vars);
            if (obj.cudas_logs.isKey(str))
                fprintf('There is a bug !!!\n');
                assert(0);
            end
            obj.cudas_logs(str) = CudasLog(info);
        end
        
        function ret = GetApproxInfo(obj, approx_vars)
            str = struct2str(approx_vars);
            if (~obj.cudas_logs.isKey(str))
                fprintf('There should be value stored\n');
                assert(0);
            end
            ret = obj.cudas_logs(str);
        end
                
        function Printf(obj)
            fprintf('Approx = %s\n', func2str(approxs{k}));
            map = maps{k};
            key_set = keys(map);
            for i = 1 : length(key_set)
                key = key_set{i};
                fprintf('\t%s ', key);
                val = map(key);
                test_error = val.test_error;
                fprintf('test_error = %f\n', test_error);
                if (~isfield(val, 'cuda_params_map'))
                    continue;
                end
                assert(0);
            end
        end
    end
    
end