classdef MonochromaticInput < Approximation
    properties        
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
            
            % If numImgColors <= 4, need filtersPerColors to be a multiple of 
            % B_Y * filtersPerThread.
            filtersPerColor = numFilters / approx_vars.numImgColors;
            if (approx_vars.numImgColors <= 4) 
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
                if (mod(filtersPerColor, B_Y)) 
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

            numImgColors = params.numImgColors;
            [assignment,colors] = litekmeans(C',numImgColors);
            colors = colors';
            %[ordsort,popo]=sort(perm);

            for f=1:size(W,1)
                chunk = (C(assignment(f),:)')*dec(f,1)*(S(f,:));
                Wapprox(f,:,:,:) = reshape(chunk,1,size(W,2),size(W,3),size(W,4));
            end

            Wapprox = permute(Wapprox, [1, 3, 4, 2]);
            Wmono = reshape(S,size(W,1),size(W,3),size(W,4));

            ret.colors =  colors;
            [~, ret.perm] = sort(assignment);
            ret.Wmono = Wmono(ret.perm, :, :);
            ret.Wapprox = Wapprox;
            ret.layer = 'ConvMono';
        end       
%         
%         function [test_error, time] = RunModifConv(obj, args)
%             global plan;
%             X = single(plan.layer{1}.cpu.vars.out);
%             colors = args.colors;
%             numImages = size(X, 1);
%             imgWidth = size(X, 2);
%             numImgColors = size(colors, 1);
%             
%             % Color transoformation of input.
%             XX = reshape(X, [numImages * imgWidth * imgWidth, 3]);
%             res = XX * colors';
%             Xmono = single(reshape(res, [numImages, imgWidth, imgWidth, numImgColors]));
%             
%             Wmono = single(args.Wmono);
%             numFilters = size(W, 1);
% 
%             perm = args.perm;
%             
%             padding = 0;
%             stride = 4;
%             
%             % Define GPU ids.
%             gids.Xmono = 1;
%             gids.Wmono = 2;
%             gids.out_mono = 3;
%             gids.perm = 4;
% 
%             % Copy to GPU.
%             Capprox_gen(CopyToGPU, gids.Wmono, Wmono);
%             Capprox_gen(CopyToGPU, gids.Xmono, Xmono);
%             Capprox_gen(CopyToGPU, gids.out_mono, out_mono_);
%             Capprox_gen(CopyToGPU, gids.perm, perm);
%             
%             ForwardPass(plan.input);
% 
%             Capprox_gen(approx_pointer, gids.Xmono, gids.Wmono, gids.out_mono, size(Xmono, 2), size(Xmono, 4), size(Wmono, 2), stride, padding, gids.perm);
%             out_mono = reshape(Capprox_gen(CopyFromGPU, gids.out_mono), size(out_mono_));
%             Capprox_gen(CleanGPU);
%             
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             % (2) Results with Wapprox should equal result with Wmono
%             % ----> ret.cuda_wapprox_eq
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             out_ = single(zeros(numImages, 55, 55, numFilters));
%             out_mono_ = single(zeros(numImages, 55, 55, numFilters));
% 
%             % copy to GPU for regular conv
%             C_(CopyToGPU, gids.Wapprox,  Wapprox);
%             C_(CopyToGPU, gids.X,  X);
%             C_(CopyToGPU, gids.out,  out_);
% 
%             C_(ConvAct, gids.X, gids.Wapprox, gids.out, size(X, 2), size(X, 4), size(Wapprox, 2), stride, padding);
%             out = reshape(C_(CopyFromGPU, gids.out), size(out_));
%             C_(CleanGPU);
% 
%             %copy to GPU for mono conv
%             Capprox_gen(CopyToGPU, gids.Wmono,  Wmono);
%             Capprox_gen(CopyToGPU, gids.Xmono,  Xmono);
%             Capprox_gen(CopyToGPU, gids.out_mono,  out_mono_);
%             Capprox_gen(CopyToGPU, gids.perm,  perm);
% 
%             Capprox_gen(approx_pointer, gids.Xmono, gids.Wmono, gids.out_mono, size(Xmono, 2), size(Xmono, 4), size(Wmono, 2), stride, padding, gids.perm);
%             out_mono = reshape(Capprox_gen(CopyFromGPU, gids.out_mono), size(out_mono_));
%             Capprox_gen(CleanGPU);
% 
%             % are results equal?
%             neq = sum(sum(sum(sum(out_mono ~= out))));
%             if neq
%                 printf(0, 'Results with Wapprox ~= results with Wmono\n');
%             end
%             ret.cuda_wapprox_eq = neq == 0;
%             
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             % (2) Check test errors with approximated results
%             % ----> ret.cuda_test_error
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             ret.cuda_test_error = 0;
% 
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             % (3) Now check the runtime of regular vs. mono version
%             % ----> ret.cuda_speedup
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             out_ = single(zeros(numImages, 55, 55, numFilters));
%             out_mono_ = single(zeros(numImages, 55, 55, numFilters));
% 
%             num_runs = 100;
% 
%             % copy to GPU for regular conv
%             C_(CopyToGPU, gids.X,  X);
%             C_(CopyToGPU, gids.W,  W);
%             C_(CopyToGPU, gids.out,  out_);
%             lapse1 = [];
%             for t=1:num_runs
%                 C_(StartTimer);
%                 C_(ConvAct, gids.X, gids.W, gids.out, size(X, 2), size(X, 4), size(W, 2), stride, padding);
%                 lapse = C_(StopTimer); 
%                 out = reshape(C_(CopyFromGPU, gids.out), size(out_));
%                 lapse1 = [lapse1, lapse];
%             end
%             C_(CleanGPU);
% 
%             % copy to GPU for mono conv
%             Capprox_gen(CopyToGPU, gids.Xmono,  Xmono);
%             Capprox_gen(CopyToGPU, gids.Wmono,  Wmono);
%             Capprox_gen(CopyToGPU, gids.out_mono,  out_mono_);
%             Capprox_gen(CopyToGPU, gids.perm,  perm);
% 
%             lapse2 = [];
%             for t=1:num_runs
%                 Capprox_gen(StartTimer);
%                 Capprox_gen(approx_pointer, gids.Xmono, gids.Wmono, gids.out_mono, size(Xmono, 2), size(Xmono, 4), size(Wmono, 2), stride, padding, gids.perm);
%                 lapse = Capprox_gen(StopTimer); 
%                 out_mono = reshape(Capprox_gen(CopyFromGPU, gids.out_mono), size(out_));
%                 lapse2 = [lapse2, lapse];
%             end
%             Capprox_gen(CleanGPU);
% 
%             speedup = lapse1 ./ lapse2;
%             ret.cuda_speedup = speedup;     
%         end
    end
    
end