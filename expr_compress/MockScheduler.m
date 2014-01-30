classdef MockScheduler < Scheduler
    properties
        compilation_counter
    end
    methods
        function obj = MockScheduler(args)
            obj@Scheduler(args);
            obj.compilation_counter = 0;
        end
        
        function Compile(obj, func_name, args, cuda_vars)
            fprintf('Mocked compilation\n');
            obj.compilation_counter = obj.compilation_counter + 1;
        end
    end
end