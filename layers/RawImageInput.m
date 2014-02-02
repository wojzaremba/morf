classdef RawImageInput < Input
    properties
        file_pattern
        meanX
        Y
    end
    methods
        function obj = RawImageInput(json)
            obj@Input(FillDefault(json));
            obj.file_pattern = json.file_pattern;
            tmp = load(sprintf('%s/meta.mat', obj.file_pattern));
            obj.meanX = tmp.meanX;
            obj.Y = tmp.Y;
            obj.Finalize();
        end       
        
        function [X, Y, batches] = LoadData(obj, file_pattern, batch_size)
            X = [];
            Y = [];
            batches = -1;
        end
        
        function [X, Y, step] = GetImage_(obj, step, train)                         
            X = zeros(obj.batch_size, obj.dims(1), obj.dims(2), 3);
            Y = zeros(obj.batch_size, 1000);
            for i = ((step - 1) * obj.batch_size + 1) : (step * obj.batch_size)
                name = sprintf('%s/ILSVRC2012_val_%s.JPEG', obj.file_pattern, sprintf('%08d', i));
                idx = i - step + 1;
                X(idx, :, :, :) = single(imread(name));
                Y(idx, obj.Y(i)) = 1;
            end    
            X = X - repmat(reshape(obj.meanX, [1, obj.dims(1), obj.dims(2), obj.dims(3)]), [obj.batch_size, 1, 1, 1]);
            step = step + 1;
        end               
    end
end

function toString(i, nr)
    zeros(nr, 1)
   num2str(i) 
end

function json = FillDefault(json)
json.type = 'RawImageInput';
end
