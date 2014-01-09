classdef Morfing < Layer
    properties
    end
    
    methods
        function obj = Morfing(json)
            obj@Layer(FillDefault(json));
            obj.Finalize();
        end
        
        function FP(layer)
            global plan
            if (plan.lr > 0)
                out = layer.params.X;
                return;
            end
            params = layer.params;
            X = params.X;
            out = X;
            d = layer.dims(2:3);
            x1 = min(max(round(squeeze(params.D(1, :, :))), 1), d(1) - 1);
            y1 = min(max(round(squeeze(params.D(2, :, :))), 1), d(2) - 1);
            x2 = x1 + 1;
            y2 = y1 + 1;
            for b = 1:size(X, 1)
                idx11 = x1 + (y1 - 1) * size(X, 3);
                idx21 = x2 + (y1 - 1) * size(X, 3);
                idx12 = x1 + (y2 - 1) * size(X, 3);
                idx22 = x2 + (y2 - 1) * size(X, 3);
                x_ = params.D(1, :);
                y_ = params.D(2, :);                
                for i = 1:layer.dims(1)
                    X_ = squeeze(X(b, i, :, :));
                    out(b, i, :) = X_(idx11(:)) .* (x2(:) - x_(:)) .* (y2(:) - y_(:)) + X_(idx21(:)) .* (x_(:) - x1(:)) .* (y2(:) - y_(:)) + X_(idx12(:)) .* (x2(:) - x_(:)) .* (y_(:) - y1(:)) + X_(idx22(:)) .* (x_(:) - x1(:)) .* (y_(:) - y1(:));
                end
            end
            layer.params.out = out;
        end
        
        function dparams = BP(layer, node_data)
            global plan
            if (plan.lr > 0)
                dparams = struct();
                return;
            end
            params = layer.params;
            X = params.X;
            D = params.D;
            dD = zeros(size(D));
            d = layer.dims(2:3);
            x1 = min(max(round(squeeze(params.D(1, :, :))), 1), d(1) - 1);
            y1 = min(max(round(squeeze(params.D(2, :, :))), 1), d(2) - 1);
            x2 = x1 + 1;
            y2 = y1 + 1;        
            jacobian = zeros(size(node_data, 1), numel(D));
            for b = 1:size(X, 1)
                idx11 = x1 + (y1 - 1) * size(X, 3);
                idx21 = x2 + (y1 - 1) * size(X, 3);
                idx12 = x1 + (y2 - 1) * size(X, 3);
                idx22 = x2 + (y2 - 1) * size(X, 3);
                x_ = params.D(1, :);
                y_ = params.D(2, :);                
                for i = 1:layer.dims(1)
                    X_ = squeeze(X(b, i, :, :));
                    dx = X_(idx11(:)) .* (-1) .* (y2(:) - y_(:)) + X_(idx21(:)) .* (y2(:) - y_(:)) + X_(idx12(:)) .* (-1) .* (y_(:) - y1(:)) + X_(idx22(:)) .* (y_(:) - y1(:));                            
                    dy = X_(idx11(:)) .* (x2(:) - x_(:)) .* (-1) + X_(idx21(:)) .* (x_(:) - x1(:)) .* (-1) + X_(idx12(:)) .* (x2(:) - x_(:)) + X_(idx22(:)) .* (x_(:) - x1(:));
                    add = [dx, dy]' .* repmat(squeeze(node_data(b, i, :))', [2, 1]);
                    dD(:, :) = dD(:, :) + add;
                    jacobian(b, :) = jacobian(b, :) + add(:)';
                end
            end
            layer.scratch.jacobian = jacobian;
            dparams = struct('dD', dD);
        end
        
        function InitWeights(obj)
            obj.params.D = zeros([2, obj.dims(2), obj.dims(3)]);
            for x = 1:obj.dims(2)
                for y = 1:obj.dims(2)
                    obj.params.D(1, x, y) = x;
                    obj.params.D(2, x, y) = y;
                end
            end
        end
    end
end

function json = FillDefault(json)
json.one2one = true;
end