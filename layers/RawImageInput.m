classdef RawImageInput < Input
    properties
        file_pattern
        train
        test
        val
        training
        full_size
        raw_images
        output_size
    end
    methods
        function obj = RawImageInput(json)
            obj@Input(FillDefault(json));
            obj.file_pattern = json.file_pattern;
            obj.raw_images = json.raw_images;
            obj.training = 1;
            obj.Finalize();
        end       
        
        function [X, Y, batches] = LoadData(obj, file_pattern, batch_size)
            X = [];
            Y = [];
            batches = -1;
        end
        
        function [X, Y, step] = GetImage_(obj, step, train)         
            if (train == 2)
                if (isempty(obj.val))
                    obj.val = struct();
                    obj.val.path = [obj.raw_images];
                    tmp = load([obj.file_pattern]);
                    obj.val.mean = repmat(reshape(single(tmp.mean_img), [1, obj.dims]), [obj.batch_size, 1, 1, 1]);
                    obj.val.Y = single(tmp.Y);
                end
                source = obj.val;            
            else
                assert(0);
            end
            X = zeros(obj.batch_size, obj.dims(1), obj.dims(2), 3);
            Y = zeros(obj.batch_size, 1000);
            for i = step : (step + obj.batch_size - 1)
                name = sprintf('%s/ILSVRC2012_val_%s.JPEG', source.path, toString(i, 8));
                X(i - step + 1, :, :, :) = single(imread(name));
                Y(i - step + 1, :) = obj.val.Y(i, :);
            end    
            X = X - source.mean;
            step = i + 1;
        end               
    end
end

function json = FillDefault(json)
json.type = 'RawImageInput';
end
