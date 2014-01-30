classdef MonoConv < LayerApprox
    properties
        num_image_colors
    end
    
    methods
        function obj = MonoConv(json)
            obj@LayerApprox(FillDefault(json));
            obj.num_image_colors = json.num_image_colors;
            obj.Finalize();
        end                      
       
        function FP_(obj)
            global plan
            bs = plan.input.batch_size;
            v = obj.gpu.vars;
            pdims = obj.prev_dim();
            Capprox_gen(approx_pointer, v.X, v.Wmono, v.out, pdims(2), obj.num_image_colors, obj.patch(1), obj.stride(1), obj.padding(1), v.perm, v.Cmono);
            Capprox_gen(Reshape, v.out, bs * obj.dims(1) * obj.dims(2), obj.depth());
            Capprox_gen(AddVector, v.out, v.B, v.out);
            Capprox_gen(Reshape, v.out, obj.dims(1) * obj.dims(2) * obj.depth(), bs);
            Capprox_gen(obj.Fun_, v.out, v.out);  
        end
        
        function FP(obj)
            prev_dim = obj.prev_dim();
            v = obj.cpu.vars;
            X = v.X;  
            bs = size(X, 1);
            % Color transform
            X = reshape(X, [bs * prev_dim(1) * prev_dim(2), prev_dim(3)]) * v.Cmono;
            X = reshape(X, [bs, prev_dim(1), prev_dim(2), obj.num_image_colors]);
            
            filters_per_color = obj.dims(3) / obj.num_image_colors;
            for c = 1 : obj.num_image_colors
                X_ = zeros(size(X, 1), prev_dim(1) + obj.padding(1) * 2 + obj.patch(1), prev_dim(2) + obj.padding(2) * 2 + obj.patch(1));
                X_(:, (obj.padding(1) + 1):(end - obj.patch(1) - obj.padding(1)), (obj.padding(2) + 1):(end - obj.patch(1) - obj.padding(2))) = X(:, :, :, c);
                stacked = zeros(size(X, 1) * prod(obj.dims(1:2)), obj.patch(1) * obj.patch(2));
                for x = 1:obj.dims(1)
                    for y = 1:obj.dims(2)
                        sx = (x - 1) * obj.stride(1) + 1;
                        ex = sx + obj.patch(1) - 1;
                        sy = (y - 1) * obj.stride(2) + 1;
                        ey = sy + obj.patch(2) - 1;
                        tmp = X_(:, sx:ex, sy:ey);
                        idx = ((y - 1) * obj.dims(1) + x - 1) * bs + 1;
                        stacked(idx : (idx + bs - 1), :) = tmp(:, :);
                    end
                end
                filt_idx = v.perm((c - 1) * filters_per_color + 1: c* filters_per_color) + 1;
                v.out(:, :, :, filt_idx) = reshape(stacked * v.Wmono(filt_idx, :)', [bs, obj.dims(1:2), filters_per_color]);
%                 fprintf('c = %d\n', c)
%                 stacked(:)
%                 v.Wmono(filt_idx, :)'
%                 fprintf('\n\n\n');
            end
            v.out = bsxfun(@plus, v.out, reshape(v.B, [1, 1, 1, length(v.B)]));
            obj.cpu.vars.forward_act = v.out;              
            obj.cpu.vars.out = obj.F(v.out);
        end     
        
        function FPcpp(obj)
            v = obj.cpu.vars; 
            MonoConvCpp(v.X, v.Xmono, v.Cmono, v.Wmono, v.B, v.perm, v.out, obj.stride(1), obj.padding(1));
        end        
        
        function BP(obj)
            assert(0);
        end
        
        function InitWeights(obj)          
            global plan
            prev_dim = obj.prev_dim();
            obj.AddParam('Xmono', [plan.input.batch_size, prev_dim(1), prev_dim(2), obj.num_image_colors], false);             
            obj.AddParam('Cmono', [prev_dim(3), obj.num_image_colors], false);            
            obj.AddParam('Wmono', [obj.depth, obj.patch(1), obj.patch(2)], false);     
            obj.AddParam('B', [obj.depth, 1], false);              
            obj.AddParam('perm', [obj.depth, 1], false); 
        end
    end
end

function json = FillDefault(json)
json.type = 'MonoConv';
end
