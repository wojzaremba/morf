classdef VariableSizeInput < Layer
    properties
        batch_size
        X
        Y
        badY        
        badX
        active_indices
        file_pattern
        lambda
    end
    methods
        function obj = VariableSizeInput(json)
            obj@Layer(json);  
            obj.batch_size = Val(json, 'batch_size', 128);
            obj.lambda = Val(json, 'lambda', 10);
            obj.file_pattern = json.file_pattern;
            global plan
            plan.input = obj;            
            load(sprintf('%s/train.mat', obj.file_pattern), 'data');
            obj.X = zeros(plan.input.batch_size, 28, 28);
            obj.Y = zeros(plan.input.batch_size, 10);
            for i = 1 : plan.input.batch_size
                obj.X(i, :, :) = data{i}.X;
                obj.Y(i, :) = data{i}.Y;
            end
            obj.active_indices = 1 : plan.input.batch_size;

            obj.badX = obj.X;
            obj.badY = zeros(plan.input.batch_size, 10);            
            obj.badY(:, 2:end) = obj.Y(:, 1:(end - 1));
            obj.badY(:, 1) = obj.Y(:, end);          
            
        end       

        function success = SetNewData(obj)
            success = (~isempty(obj.active_indices)) && (obj.lambda > 1e-3);
            fprintf(1, 'Num actives: %d\n', length(obj.active_indices(:)));
            
            obj.cpu.vars.out = obj.badX(obj.active_indices, :, :);
            obj.cpu.vars.Y = obj.badY(obj.active_indices, :, :);
            obj.batch_size = length(obj.active_indices(:));
        end
        
        function FP_(obj) 
        end        
        
        function FP(obj) 
        end
        
        function BP_(~, ~)
        end        
        
        function BP(~, ~)
        end                
    end
end

