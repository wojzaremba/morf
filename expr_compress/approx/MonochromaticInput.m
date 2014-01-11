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
        end
        
        function ret = RunCuda(obj, args)
            global plan;
            X = single(plan.layer{1}.cpu.vars.out);
            W = single(plan.layer{2}.cpu.vars.W);
            colors = args.colors;
            numImages = size(X, 1);
            imgWidth = size(X, 2);
            numImgColors = size(colors, 1);
            XX = reshape(X, [numImages*imgWidth*imgWidth, 3]);
            res = XX * colors';
            Xmono = single(reshape(res, [numImages, imgWidth, imgWidth, numImgColors]));

            Wmono = single(args.Wmono);
            Wapprox = single(args.Wapprox);
            numFilters = size(W, 1);

            perm = args.perm;
            
            padding = 0;
            stride = 4;
            
            % Define GPU ids
            gids.X = 1;
            gids.Xmono = 2;
            gids.W = 3;
            gids.Wmono = 4;
            gids.Wapprox = 5;
            gids.out = 6;
            gids.out_mono = 7;
            gids.perm = 8;

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % (1) First check correctness of cuda code: results with 
            % Wapprox should equal result with Wmono
            % ----> ret.cuda_eq
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            out_ = single(zeros(numImages, 55, 55, numFilters));
            out_mono_ = single(zeros(numImages, 55, 55, numFilters));

            % copy to GPU for regular conv
            C_(CopyToGPU, gids.Wapprox,  h);
            C_(CopyToGPU, gids.X,  X);
            C_(CopyToGPU, gids.out,  out_);

            C_(ConvAct, gids.X, gids.Wapprox, gids.out, size(X, 2), size(X, 4), size(Wapprox, 2), stride, padding);
            out = reshape(C_(CopyFromGPU, gids.out), size(out_));
            C_(CleanGPU);

            %copy to GPU for mono conv
            Capprox_gen(CopyToGPU, gids.Wmono,  Wmono);
            Capprox_gen(CopyToGPU, gids.Xmono,  Xmono);
            Capprox_gen(CopyToGPU, gids.out_mono,  out_mono_);
            Capprox_gen(CopyToGPU, gids.perm,  perm);

            Capprox_gen(monochromatic_input, gids.Xmono, gids.Wmono, gids.out_mono, size(Xmono, 2), size(Xmono, 4), size(Wmono, 2), stride, padding, gids.perm);
            out_mono = reshape(Capprox_gen(CopyFromGPU, gids.out_mono), size(out_mono_));
            Capprox_gen(CleanGPU);

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
            Capprox_gen(CopyToGPU, gids.Xmono,  Xmono);
            Capprox_gen(CopyToGPU, gids.Wmono,  Wmono);
            Capprox_gen(CopyToGPU, gids.out_mono,  out_mono_);
            Capprox_gen(CopyToGPU, gids.perm,  perm);

            lapse2 = [];
            for t=1:num_runs
                Capprox_gen(StartTimer);
                Capprox_gen(monochromatic_input, gids.Xmono, gids.Wmono, gids.out_mono, size(Xmono, 2), size(Xmono, 4), size(Wmono, 2), stride, padding, gids.perm);
                lapse = Capprox_gen(StopTimer); 
                out_mono = reshape(Capprox_gen(CopyFromGPU, gids.out_mono), size(out_));
                lapse2 = [lapse2, lapse];
            end
            Capprox_gen(CleanGPU);

            speedup = lapse1 ./ lapse2;
            ret.cuda_speedup = speedup;     
        end
    end
    
end