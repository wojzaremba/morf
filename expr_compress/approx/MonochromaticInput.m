classdef MonochromaticInput < Approximation
    properties        
    end
    
    methods
        function obj = MonochromaticInput(approx_vars, cuda_vars)
            obj@Approximation(approx_vars, cuda_vars);
            obj.name = 'monochromatic_input';
        end
        
        function ret = VerifyCombination(obj, approx_vars, cuda_vars)
            ret = true;
            % XXX : temporarly assert(0);
%             assert(0);
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

            numImgColors = params.numImgColors;
            [perm,colors] = litekmeans(C',numImgColors);
            colors = colors';
            %[ordsort,popo]=sort(perm);

            for f=1:size(W,1)
                chunk = (C(perm(f),:)')*dec(f,1)*(S(f,:));
                Wapprox(f,:,:,:) = reshape(chunk,1,size(W,2),size(W,3),size(W,4));
            end

            Wapprox = permute(Wapprox, [1, 3, 4, 2]);
            Wmono = reshape(S,size(W,1),size(W,3),size(W,4));

            ret.colors =  colors;
            ret.perm = perm;
            ret.Wmono = Wmono;
            ret.Wapprox = Wapprox;
            ret.cuda_template = strcat(root_path, 'cuda/src/filter_acts_mono_template.cuh');
            ret.cuda_true = strcat(root_path, 'cuda/src/filter_acts_mono.cuh');     
        end
        
        function ret = RunCuda(obj, args)
            global plan;
            X = plan.layer{1}.cpu.vars.X;
            W = plan.layer{2}.cpu.vars.W;
            numImages = size(X, 1);
            imgWidth = size(X, 2);
            numImgColors = size(colors, 1);
            XX = reshape(X, [numImages*imgWidth*imgWidth, 3]);
            res = XX * colors';
            Xmono = reshape(res, [numImages, imgWidth, imgWidth, numImgColors]);

            Wmono = args.Wmono;
            Wapprox = args.Wapprox;
            numFilters = size(W, 1);

            perm = args.perm;

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % (1) First check correctness of cuda code: results with 
            % Wapprox should equal result with Wmono
            % ----> ret.cuda_eq
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            out_ = single(zeros(numImages, 55, 55, numFilters));
            out_mono_ = single(zeros(numImages, 55, 55, numFilters));

            % copy to GPU for regular conv
            C_(CopyToGPU, gids.Wapprox,  Wapprox);
            C_(CopyToGPU, gids.X,  X);
            C_(CopyToGPU, gids.out,  out_);

            C_(ConvAct, gids.X, gids.Wapprox, gids.out, size(X, 2), size(X, 4), size(Wapprox, 2), stride, padding);
            out = reshape(C_(CopyFromGPU, gids.out), size(out_));
            C_(CleanGPU);


            %copy to GPU for mono conv
            Capprox_(CopyToGPU, gids.Wmono,  Wmono);
            Capprox_(CopyToGPU, gids.Xmono,  Xmono);
            Capprox_(CopyToGPU, gids.out_mono,  out_mono_);
            Capprox_(CopyToGPU, gids.perm,  perm);

            Capprox_(ConvActMono, gids.Xmono, gids.Wmono, gids.out_mono, size(Xmono, 2), size(Xmono, 4), size(Wmono, 2), stride, padding, gids.perm);
            out_mono = reshape(Capprox_(CopyFromGPU, gids.out_mono), size(out_mono_));
            Capprox_(CleanGPU);

            % are results equal?
            eq = sum(sum(sum(sum(out_mono ~= out))));
            if eq
                fprintf('Monochromatic conv output is incorrect\n');
            end
            ret.cuda_eq = eq == 0;
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % (2) Check test errors with approximated results
            % ----> ret.cuda_test_error
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % (3) Now check the runtime of regular vs. mono version
            % ----> ret.cuda_speedup
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            out_ = single(zeros(numImages, 55, 55, numFilters));
            out_mono_ = single(zeros(numImages, 55, 55, numFilters));

            num_runs = 100;

            % copy to GPU for regular conv
            C_(CopyToGPU, gids.X,  X);
            C_(CopyToGPU, gids.W,  W);
            C_(CopyToGPU, gids.out,  out_);
            lapse1 = [];
            for t=1:num_runs
                C_(StartTimer);
                C_(ConvAct, gids.X, gids.W, gids.out, size(X, 2), size(X, 4), size(W, 2), stride, padding);
                lapse = C_(StopTimer); 
                out = reshape(C_(CopyFromGPU, gids.out), size(out_));
                lapse1 = [lapse1, lapse];
            end
            C_(CleanGPU);

            % copy to GPU for mono conv
            Capprox_(CopyToGPU, gids.Xmono,  Xmono);
            Capprox_(CopyToGPU, gids.Wmono,  Wmono);
            Capprox_(CopyToGPU, gids.out_mono,  out_mono_);
            Capprox_(CopyToGPU, gids.perm,  perm);

            lapse2 = [];
            for t=1:num_runs
                Capprox_(StartTimer);
                Capprox_(monochromatic_input, gids.Xmono, gids.Wmono, gids.out_mono, size(Xmono, 2), size(Xmono, 4), size(Wmono, 2), stride, padding, gids.perm);
                lapse = Capprox_(StopTimer); 
                out_mono = reshape(Capprox_(CopyFromGPU, gids.out_mono), size(out_));
                lapse2 = [lapse2, lapse];
            end
            Capprox_(CleanGPU);

            speedup = lapse1 ./ lapse2;
            ret.cuda_speedup = speedup;     
        end
    end
    
end