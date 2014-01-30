classdef Softmax < Layer
    properties
        beta
    end
    
    methods
        function obj = Softmax(json)
            obj@Layer(FillDefault(json));
            obj.beta = Val(json, 'beta', 1);            
            obj.Finalize();
            global plan
            plan.classifier = obj;
            pdims = obj.prev_dim;
            assert(pdims(1) == 1);
            assert(pdims(2) == 1);
            assert(pdims(3) > 1);
        end
        
        function FP_(obj)
            assert(obj.beta == 1);
            v = obj.gpu.vars;
            C_(Max, v.X, 1, v.max);
            C_(AddVector, v.X, v.max, v.pred);
            C_(ActEXP, v.pred, v.pred);            
            C_(Sum, v.pred, 1, v.sum);
            C_(EltwiseDivideByVector, v.pred, v.sum);
            % missing part to generate out.
        end
        
        function BP_(obj)
            global plan           
            if (plan.input.on_gpu == 0)
                C_(CopyToGPU, plan.input.gpu.vars.Y, single(plan.input.cpu.vars.Y(:, :)));
            end
            C_(Subtract, obj.gpu.vars.pred, plan.input.gpu.vars.Y, obj.gpu.dvars.X);
        end

        function FP(obj)
            global plan
            X = obj.cpu.vars.X(:, :);
            shift = max(X, [], 2);
            X = X - repmat(shift, 1, prod(obj.prev_dim()));
            X = exp(plan.tiny + X ./ obj.beta);
            pred = X ./ repmat(sum(X, 2), 1, size(X, 2));
            obj.cpu.vars.pred = reshape(pred, size(X));
            obj.cpu.vars.out = -log(pred(logical(plan.input.cpu.vars.Y)));
        end
        
        function BP(obj)
            global plan
            obj.cpu.dvars.X = (obj.cpu.vars.pred - plan.input.cpu.vars.Y) / obj.beta; 
        end
        
        function InitWeights(obj)
            global plan
            bs = plan.input.batch_size;
            obj.AddParam('out', bs, false);
            obj.AddParam('pred',[bs, prod(obj.prev_dim)], false);             
            obj.AddParam('max', [bs, prod(obj.prev_dim)], false);   
            obj.AddParam('sum', [bs, prod(obj.prev_dim)], false);   
        end        
        
        function incorrect = GetScore(obj, top)
            global plan
            if (~obj.on_gpu)
                X = obj.cpu.vars.X;
            else
                X = C_(CopyFromGPU, obj.gpu.vars.X);
            end
            if (~exist('top', 'var'))
                top = 1;
            end
            correct = 0;
            assert(size(plan.input.cpu.vars.Y, 1) == size(X, 1));
            assert(size(plan.input.cpu.vars.Y, 2) == size(X, 2));
            for i = 1:plan.input.batch_size
                [~, idx] = sort(X(i, :), 'descend');    
                for j = 1:top
                    Y = plan.input.cpu.vars.Y(i, :);
                    if (idx(j) == find(Y(:)))
                        correct = correct + 1;
                    end
                end
            end            
            incorrect = plan.input.batch_size - correct;
        end        
        
        function cost = Cost(obj)
            global plan
            gt = plan.input.cpu.vars.Y(:, :);
            pred = obj.cpu.vars.pred(:, :);
            cost = -sum(sum(gt .* log(pred))) / size(obj.cpu.vars.pred, 1);            
        end
        
        function acc = GetAcc(obj)
            score = obj.GetScore();
            input_size = size(obj.cpu.vars.X, 1);
            acc = 100 * (input_size - score) / input_size;            
        end
        
        function active_indices = CorrectIndices(obj)
            global plan
            J = obj.cpu.vars.pred';
            [pred_labels, ~] = find(repmat(max(J), size(J, 1), 1) == J);
            [~, Y] = max(plan.input.Y, [], 2);
            active_indices = find(pred_labels == Y);
        end
        
    end
end

function json = FillDefault(json)
    json.type = 'Softmax';
end