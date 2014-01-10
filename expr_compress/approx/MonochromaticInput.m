classdef MonochromaticInput < Approximation
    properties        
    end
    
    methods
        function obj = MonochromaticInput(func_vars, template_vars)
            obj@Approximation(func_vars, template_vars);
            obj.name = 'monochromatic_input';
        end
        
        function ret = VerifyCombination(obj, func_vars, template_vars)
            ret = true;
            % XXX : temporarly assert(0);
%             assert(0);
        end        
        
        function [Wapprox, args] = Approx(obj, params)
            Wapprox = [];
            args = [];
            
            %global plan;
            X = randn(128, 224, 224, 3); %plan.layer{1}.cpu.vars.X;
            W = randn(96, 3, 11, 11); %permute(params.W, [1, 4, 2, 3]);
            for f=1:size(W, 1)
                [u,s,v] = svd(squeeze(W(f,:,:)),0);
                C(f,:) = u(:,1);
                S(f,:) = v(:,1);
                dec(f,:) = diag(s);
            end

            numImgColors = params.numImgColors;
            [perm,colors] = litekmeans(C',numImgColors);
            colors = colors';
            [ordsort,popo]=sort(perm);

            for f=1:size(W,1)
                chunk = (C(perm(f),:)')*dec(f,1)*(S(f,:));
                Wapprox(f,:,:,:) = reshape(chunk,1,size(W,2),size(W,3),size(W,4));
            end

            Wapprox = permute(Wapprox, [1, 3, 4, 2]);
            Wmono = reshape(S,size(W,1),size(W,3),size(W,4));

            args = struct('colors', colors, 'Wmono', Wmono, 'perm', perm);
            args.cuda_template = 'cuda/src/filter_acts_mono_template.cuh';
            args.cuda_true = 'cuda/src/filter_acts_mono.cuh'; 
            args.X = X;
            args.W = W;
            args.Wmono = Wmono;
            args.Wapprox = Wapprox;
            args.colors = colors;
            args.perm = perm;    
        end
    end
    
end