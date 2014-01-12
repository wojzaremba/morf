classdef NoApproximation < Approximation
    properties        
    end
    
    methods
        function obj = NoApproximation(suffix, approx_vars, cuda_vars)
            obj@Approximation(suffix, approx_vars, cuda_vars);
            obj.name = 'no_approximation';
        end
        
        function ret = VerifyCombination(obj, approx_vars, cuda_vars)
            ret = true;
        end        
        
        function [Wapprox, ret] = Approx(obj, params)            
            global plan;
            Wapprox = plan.layer{2}.cpu.vars.W;          
            ret.Wapprox = Wapprox;
        end
        
        function ret = RunCuda(obj, args)
            assert(0);
            global plan;
            X = single(plan.layer{1}.cpu.vars.out);
            W = single(plan.layer{2}.cpu.vars.W);
            colors = args.colors;
            numImages = size(X, 1);
            imgWidth = size(X, 2);
            numImgColors = size(colors, 1);
            
            % Color transoformation of input.
            XX = reshape(X, [numImages * imgWidth * imgWidth, 3]);
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
            % (1) First check correctness of cuda code
            % ----> ret.cuda_correct
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            out_ = single(zeros(numImages, 55, 55, numFilters));
            out_mono_ = single(zeros(numImages, 55, 55, numFilters));

            W = single(zeros(numFilters, 11, 11, numImgColors));
            bpt = 0:(numFilters/numImgColors):numFilters;
            for i=1:numImgColors
               W( bpt(i)+1:bpt(i+1), :, :, i) = Wmono( bpt(i)+1:bpt(i+1), :, : );
            end

            % copy to GPU for regular conv
            C_(CopyToGPU, gids.W,  W);
            C_(CopyToGPU, gids.X,  X);
            C_(CopyToGPU, gids.out,  out_);

            C_(ConvAct, gids.X, gids.W, gids.out, size(X, 2), size(X, 4), size(W, 2), stride, padding);
            out = reshape(C_(CopyFromGPU, gids.out), size(out_));
            C_(CleanGPU);

            %copy to GPU for mono conv
            Capprox_gen(CopyToGPU, gids.Wmono,  Wmono);
            Capprox_gen(CopyToGPU, gids.X,  X);
            Capprox_gen(CopyToGPU, gids.out_mono,  out_mono_);
            Capprox_gen(CopyToGPU, gids.perm,  perm);

            Capprox_gen(approx_pointer, gids.X, gids.Wmono, gids.out_mono, size(X, 2), size(X, 4), size(Wmono, 2), stride, padding, gids.perm);
            out_mono = reshape(Capprox_gen(CopyFromGPU, gids.out_mono), size(out_mono_));
            Capprox_gen(CleanGPU);

            % are results equal?
            neq = sum(sum(sum(sum(out_mono ~= out))));
            if neq
                printf(0, 'Monochromatic conv output is incorrect\n');
            end
            ret.cuda_correct = neq == 0;
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % (2) Results with Wapprox should equal result with Wmono
            % ----> ret.cuda_wapprox_eq
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
            Capprox_gen(CopyToGPU, gids.Wmono,  Wmono);
            Capprox_gen(CopyToGPU, gids.Xmono,  Xmono);
            Capprox_gen(CopyToGPU, gids.out_mono,  out_mono_);
            Capprox_gen(CopyToGPU, gids.perm,  perm);

            Capprox_gen(approx_pointer, gids.Xmono, gids.Wmono, gids.out_mono, size(Xmono, 2), size(Xmono, 4), size(Wmono, 2), stride, padding, gids.perm);
            out_mono = reshape(Capprox_gen(CopyFromGPU, gids.out_mono), size(out_mono_));
            Capprox_gen(CleanGPU);

            % are results equal?
            neq = sum(sum(sum(sum(out_mono ~= out))));
            if neq
                printf(0, 'Results with Wapprox ~= results with Wmono\n');
            end
            ret.cuda_wapprox_eq = neq == 0;
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % (2) Check test errors with approximated results
            % ----> ret.cuda_test_error
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            ret.cuda_test_error = 0;

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
                Capprox_gen(approx_pointer, gids.Xmono, gids.Wmono, gids.out_mono, size(Xmono, 2), size(Xmono, 4), size(Wmono, 2), stride, padding, gids.perm);
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