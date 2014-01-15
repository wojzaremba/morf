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
            Capprox_gen(Reshape, v.X, pdims(3), bs * pdims(1) * pdims(2));
            Capprox_gen(Mult, v.X, v.Cmono, v.Xmono);
            Capprox_gen(Reshape, v.Xmono, pdims(1) * pdims(2) * obj.num_image_colors, bs);
            Capprox_gen(approx_pointer, v.Xmono, v.Wmono, v.out, pdims(2), obj.num_image_colors, obj.patch(1), obj.stride(1), obj.padding(1), v.perm);
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
            obj.cpu.vars.X_ = zeros(size(X, 1), prev_dim(1) + obj.padding(1) * 2 + obj.patch(1), prev_dim(2) + obj.padding(2) * 2 + obj.patch(1), prev_dim(3));
            obj.cpu.vars.X_(:, (obj.padding(1) + 1):(end - obj.patch(1) - obj.padding(1)), (obj.padding(2) + 1):(end - obj.patch(1) - obj.padding(2)), :) = X;
            obj.cpu.vars.stacked = zeros(size(X, 1) * prod(obj.dims(1:2)), obj.patch(1) * obj.patch(2) * prev_dim(3));
            for x = 1:obj.dims(1)
                for y = 1:obj.dims(2)
                    sx = (x - 1) * obj.stride(1) + 1;
                    ex = sx + obj.patch(1) - 1;
                    sy = (y - 1) * obj.stride(2) + 1;
                    ey = sy + obj.patch(2) - 1;
                    tmp = obj.cpu.vars.X_(:, sx:ex, sy:ey, :);
                    idx = ((y - 1) * obj.dims(1) + x - 1) * bs + 1;
                    obj.cpu.vars.stacked(idx : (idx + bs - 1), :) = tmp(:, :);
                end
            end
            results = reshape(obj.cpu.vars.stacked * v.W(:, :)', [bs, obj.dims(1:2), obj.depth]);           
            results = bsxfun(@plus, results, reshape(v.B, [1, 1, 1, length(v.B)]));
            obj.cpu.vars.forward_act = results;              
            obj.cpu.vars.out = obj.F(results);
        end     
        
        function BP(obj)
            assert(0);
        end
        
        % XXX: allocate forward_act var and other missing.
        function InitWeights(obj)          
            global plan
            prev_dim = obj.prev_dim();
            obj.AddParam('B', [obj.depth, 1], false);  
            obj.AddParam('Cmono', [obj.num_image_colors, prev_dim(3)], false);            
            obj.AddParam('Wmono', [obj.depth, obj.patch(1), obj.patch(2)], false);     
            obj.AddParam('perm', [obj.depth], false); 
            obj.AddParam('Xmono', [obj.prev{1}.batch_size, prev_dim(1), prev_dim(2), prev_dim(3)], false); 
        end
    end
end

function json = FillDefault(json)
json.type = 'MonoConv';
end
