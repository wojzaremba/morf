classdef TestInput < Input
    properties
        number_of_classes
    end
    methods
        function obj = TestInput(json)
            obj@Input(FillDefault(json));
            obj.number_of_classes = Val(json, 'number_of_classes', 4);
            obj.Finalize();
        end
        
        function [X, Y, step] = GetImage_(obj, step, train)
            randn('seed', step);
            rand('seed', step);
            X = 1 * randn([obj.batch_size, obj.dims]);
            Y = zeros(obj.batch_size, obj.number_of_classes);
            for i = 1:obj.batch_size
                Y(i, randi(obj.number_of_classes)) = 1;
            end
            step = 1;
        end
        
        function ReloadData(obj, batch_size)
        end
    end
end


function json = FillDefault(json)
json.type = 'TestInput';
end