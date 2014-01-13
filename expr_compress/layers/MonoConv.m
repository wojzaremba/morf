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
            v = obj.gpu.vars;           
            dims = obj.dims;
            pdims = obj.prev_dim();     
            bs = obj.prev{1}.batch_size;
            %Capprox_gen(Mult, v.X, v.Cmono, v.Xmono);
            %XXX : do transformation on GPU
            X = Capprox_gen(CopyFromGPU, v.X); % This is redundant because ForwardPass just copied it onto the GPU, and we're removing it without doing any work
            X = reshape(X, bs*pdims(1)*pdims(2), pdims(3));
            Capprox_gen(CopyToGPU, v.X,  single(X));
            Capprox_gen(Mult, v.X, v.Cmono, v.Xmono);
            Xmono = Capprox_gen(CopyFromGPU, v.Xmono);
            Xmono = reshape(Xmono, bs, pdims(1), pdims(2), obj.num_image_colors);
            Capprox_gen(CopyToGPU, v.Xmono,  single(Xmono));
            %Capprox_gen(Reshape, v.Xmono, pdims(1)*pdims(2)*obj.num_image_colors, bs);
            Capprox_gen(approx_pointer, v.Xmono, v.Wmono, v.out, pdims(2), obj.num_image_colors, obj.patch(1), obj.stride(1), obj.padding(1), v.perm);
            Capprox_gen(Reshape, v.out, bs * obj.dims(1) * obj.dims(2), obj.depth());
            Capprox_gen(AddVector, v.out, v.B, v.out);
            Capprox_gen(Reshape, v.out, obj.dims(1) * obj.dims(2) * obj.depth(), bs);
            Capprox_gen(obj.Fun_, v.out, v.out);      
            out = Capprox_gen(CopyFromGPU, v.out);
            out = reshape(out, [bs, dims(1), dims(2), dims(3)]);
            perm = Capprox_gen(CopyFromGPU, v.perm);
            out = out(:, :, :, perm);
            Capprox_gen(CopyToGPU, v.out, single(out));
        end
        
        function FP(obj)
            assert(0);
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
