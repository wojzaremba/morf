function ret = monochromatic_input_cuda_interface(args)
    
    X = args.X;
    numImages = size(X, 1);
    imgWidth = size(X, 2);
    numImages = size(X, 1);
    numImgColors = size(colors, 1);
    XX = reshape(X, [numImages*imgWidth*imgWidth, 3]);
    res = XX * colors';
    Xmono = reshape(res, [numImages, imgWidth, imgWidth, numImgColors]);
    
    W = args.W;
    Wmono = args.Wmono;
    Wapprox = args.Wapprox;
    numFilters = size(W, 1);
    
    perm = args.perm;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % (1) First check correctness of cuda code: results with Wapprox should
    % equal result with Wmono
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
    Cmono_(CopyToGPU, gids.Wmono,  Wmono);
    Cmono_(CopyToGPU, gids.Xmono,  Xmono);
    Cmono_(CopyToGPU, gids.out_mono,  out_mono_);
    Cmono_(CopyToGPU, gids.perm,  perm);

    Cmono_(ConvActMono, gids.Xmono, gids.Wmono, gids.out_mono, size(Xmono, 2), size(Xmono, 4), size(Wmono, 2), stride, padding, gids.perm);
    out_mono = reshape(Cmono_(CopyFromGPU, gids.out_mono), size(out_mono_));
    Cmono_(CleanGPU);

    % are results equal?
    eq = sum(sum(sum(sum(out_mono ~= out))));
    if eq
        fprintf('Monochromatic conv output is incorrect\n');
    end 

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % (2) Check test errors with approximated results
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % (3) Now check the runtime of regular vs. mono version
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
    Cmono_(CopyToGPU, gids.Xmono,  Xmono);
    Cmono_(CopyToGPU, gids.Wmono,  Wmono);
    Cmono_(CopyToGPU, gids.out_mono,  out_mono_);
    Cmono_(CopyToGPU, gids.perm,  perm);

    lapse2 = [];
    for t=1:num_runs
        Cmono_(StartTimer);
        Cmono_(ConvActMono, gids.Xmono, gids.Wmono, gids.out_mono, size(Xmono, 2), size(Xmono, 4), size(Wmono, 2), stride, padding, gids.perm);
        lapse = Cmono_(StopTimer); 
        out_mono = reshape(Cmono_(CopyFromGPU, gids.out_mono), size(out_));
        lapse2 = [lapse2, lapse];
    end
    Cmono_(CleanGPU);

    speedup = lapse1 ./ lapse2;
%     fprintf('average speedup = %f\n', mean(speedup));
%     fprintf('std speedup = %f\n', std(speedup));

end