classdef ConvFFT < Layer
    properties
    end
    
    methods
        function obj = ConvFFT(json)
            obj@Layer(FillDefault(json));
            obj.Finalize();
        end                      
                
        function FP(obj)
            params = obj.params;          
            X = obj.params.X;

            W_ = obj.params.W_;
            out = zeros([size(X, 1), size(W_, 1), size(W_, 2), size(params.W, 1)]);
            for b = 1:size(X, 1)
                out(b, :, :, :) = real(ifft2(squeeze(sum(bsxfun(@times, fft2(squeeze(X(b, :, :, :)), size(W_, 1), size(W_, 2)), W_), 3))));
            end
            pad = floor(size(obj.params.W, 2) / 2);
            out = out(:, (pad + 1):1:(end - pad), (pad + 1):1:(end - pad), :, :);
            
            out = out(:, 1:obj.stride(1):end, 1:obj.stride(2):end, :);
            out = out(:, 1:obj.dims(1), 1:obj.dims(2), :);
            
            out = bsxfun(@plus, out, reshape(obj.params.B, [1, 1, 1, length(obj.params.B)]));
            obj.params.out = obj.F(out);            
        end    
        
        function InitWeights(obj)
            prev_dim = obj.prev_dim();
            obj.AddParam('W', [obj.depth, obj.patch(1), obj.patch(2), prev_dim(3)], true);
            obj.AddParam('B', [obj.depth, 1], true);            
        end
    end
end

function json = FillDefault(json)
json.type = 'ConvFFT';
end
