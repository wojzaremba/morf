classdef CudasLog < handle
    properties
        test_error
        
    end
    
    % XXX : get a gpu name.
    
    methods
        function obj = CudasLog(info)            
            obj.test_error = info.test_error;            
        end
        
        function AddCudaExecution(obj, cuda_params, speed)
            assert(0);            
        end
        
        function Printf(obj)
            assert(0);
        end
        
        function BestSpeed(obj)
            assert(0);
        end
        
    end
    
end