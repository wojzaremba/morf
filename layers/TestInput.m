classdef TestInput < Input
    properties
    end
    methods
        function obj = TestInput(json)
            obj@Input(FillDefault(json));
            obj.Finalize();
        end
        
        function [X, Y, step] = GetImage_(obj, step, train)
            randn('seed', step);
            rand('seed', step);
            X = 1 * randn([obj.batch_size, obj.dims]);
            Y = zeros(obj.batch_size, obj.dims(3));
            for i = 1:obj.batch_size
                Y(i, randi(obj.dims(3))) = 1;
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