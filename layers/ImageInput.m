classdef ImageInput < Input
    properties
        file_pattern
        train
        test
        val
        training
        output_size
    end
    methods
        function obj = ImageInput(json)
            obj@Input(FillDefault(json));
            obj.file_pattern = json.file_pattern;
            obj.training = 1;
            obj.Finalize();
        end       
        
        function [X, Y, batches] = LoadData(obj, file_pattern, batch_size)
            try 
                load(sprintf('%s%d', file_pattern, batch_size), 'X', 'Y', 'batches');
                return;
            catch
            end
            tmp = load(file_pattern);
            data = tmp.data;
            obj.output_size = size(data{1}.Y);
            batches = floor(length(data) / batch_size);
            real_input_dim = size(data{1}.X);
            X = cell(batches, 1);
            Y = cell(batches, 1);
            for step = 1:batches
                X_tmp = zeros([batch_size, real_input_dim]);
                Y_tmp = zeros([batch_size, obj.output_size]);
                for j = 1:batch_size
                    datapoint = data{(step - 1) * batch_size + j};
                    X_tmp(j, :) = datapoint.X(:);
                    Y_tmp(j, :) = datapoint.Y(:);
                end
                X{step} = X_tmp;
                Y{step} = Y_tmp;
            end            
            prefix = '';
            if (length(dir('data')) == 0)
                prefix = '../';
            end
            save(sprintf('%s%s%d', prefix, file_pattern, batch_size), 'X', 'Y', 'batches');            
        end
        
        function [X, Y, step] = GetImage_(obj, step, train)
            X = [];
            Y = [];
            if (train == 2)
                if (isempty(obj.val))
                    [obj.val.X, obj.val.Y, obj.val.batches] = ...
                    obj.LoadData([obj.file_pattern, 'val'], obj.batch_size);
                    obj.val.maxbatches = Inf;
                end
                source = obj.val;            
            elseif (train == 1)
                if (isempty(obj.train))
                    [obj.train.X, obj.train.Y, obj.train.batches] = ...
                    obj.LoadData([obj.file_pattern, 'train'], obj.batch_size);
                    obj.train.maxbatches = Inf;
                end
                source = obj.train;
            elseif (train == 0) 
                if (isempty(obj.test))
                    [obj.test.X, obj.test.Y, obj.test.batches] = ...
                    obj.LoadData([obj.file_pattern, 'test'], obj.batch_size);
                    obj.test.maxbatches = Inf;
                end
                source = obj.test;
            end
            if (step > min(source.batches, source.maxbatches))
                step = -1;
                return;
            end
            X = source.X{step};
            Y = source.Y{step};            
            step = step + 1;
        end               
    end
end

function json = FillDefault(json)
json.type = 'ImageInput';
end
