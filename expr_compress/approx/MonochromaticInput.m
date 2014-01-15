classdef MonochromaticInput < Approximation
    properties        
    end
    
    methods(Static)
        function Wapprox = ReconstructW(colors, dec, S, assignment, dims)
            % dims order : (num_filters, num_colors, X, Y)
            Wapprox = zeros(dims);
            for f=1 : dims(1)
                chunk = (colors(assignment(f),:)')*dec(f,1)*(S(f,:));
                Wapprox(f, :, :, :) = reshape(chunk, [1, dims(2), dims(3), dims(4)]);
            end            
            % Rearrange so dims are : (num_filters, X, Y, num_colors) which
            % is what plan layers expects.
            Wapprox = permute(Wapprox, [1, 3, 4, 2]);
        end
    end
    
    methods
        function obj = MonochromaticInput(suffix, approx_vars, cuda_vars)
            obj@Approximation(suffix, approx_vars, cuda_vars);
            obj.name = 'monochromatic_input';
        end
        
        function ret = VerifyCombination(obj, approx_vars, cuda_vars)
            global plan;
            ret = true;
            numImages = plan.layer{1}.batch_size;
            numFilters = plan.layer{2}.dims(3);
            
            % Scale should be 0
            if (cuda_vars.scale ~= 0) 
                ret = false;
            end
            
            % Make sure checkImgBounds is correct
            if (cuda_vars.checkImgBounds ~= mod(numImages, cuda_vars.B_X*cuda_vars.imgsPerThread) )
                ret = false;
            end
            
            % If num_image_colors <= 4, need filtersPerColors to be a multiple of 
            % B_Y * filtersPerThread.
            filtersPerColor = numFilters / approx_vars.num_image_colors;
            if (approx_vars.num_image_colors <= 4) 
                if (mod(filtersPerColor, cuda_vars.B_Y * cuda_vars.filtersPerThread) )
                    ret = false;
                end
                if (cuda_vars.B_Y * cuda_vars.filtersPerThread > filtersPerColor) 
                    if (cuda_vars.B_Y * cuda_vars.filtersPerThread / filtersPerColor ~= 2) 
                        ret = false;
                    end
                       
                    if (cuda_vars.B_Y * cuda_vars.filtersPerThread / filtersPerColor == 2 && cuda_vars.colorsPerBlock ~= 2)
                       ret = false;
                    end                    
                end
            else
                if (mod(filtersPerColor, cuda_vars.B_Y)) 
                    ret = false;
                end                
            end
        end        
        
        function [Wapprox, ret] = Approx(obj, params)            
            global plan;
            global root_path;
            W = permute(plan.layer{2}.cpu.vars.W, [1, 4, 2, 3]);
            for f=1:size(W, 1)
                [u,s,v] = svd(squeeze(W(f,:,:)),0);
                C(f,:) = u(:,1);
                S(f,:) = v(:,1);
                dec(f,:) = diag(s);
            end

            num_image_colors = params.num_image_colors;
            [assignment,colors] = litekmeans(C',num_image_colors);
            colors = colors';
            % Permutaiton of W, back to orig form is done inside
            % reconstruction function.
            Wapprox = MonochromaticInput.ReconstructW(colors, dec, S, assignment, size(W));
            Wmono = reshape(bsxfun(@times, S, dec(:, 1)), size(W,1), size(W,3), size(W,4));
            
            assert(norm(squeeze(Wmono(1, :, :)) * colors(assignment(1), 1) + ...
                        squeeze(Wmono(1, :, :)) * colors(assignment(1), 2) + ...
                        squeeze(Wmono(1, :, :)) * colors(assignment(1), 3) - ...
                        (squeeze(Wapprox(1, :, :, 1)) + squeeze(Wapprox(1, :, :, 2)) + squeeze(Wapprox(1, :, :, 3)))) <= 1e-10)
        
            [~, perm] = sort(assignment);
            ret.reconstruction_error = norm(Wapprox(:) - W(:)) / norm(W(:));
            ret.vars.Wmono = Wmono;%(perm, :, :);
            ret.vars.Cmono = colors';
            ret.vars.perm = perm - 1; % Need -1 for indexing in C
            ret.Wapprox = Wapprox;
            ret.layer = 'MonoConv';
            ret.layer_nr = 2;
            ret.json = struct('num_image_colors', num_image_colors);
            ret.on_gpu = 0;
        end       

    end
    
end