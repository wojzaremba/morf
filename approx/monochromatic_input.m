function [Wapprox, args] = monochromatic_input(params)
    global morf_path;
    
    X = params.X;
    W = permute(params.W, [1, 4, 2, 3]);
    for f=1:size(W, 1)
        [u,s,v] = svd(squeeze(W(f,:,:)),0);
        C(f,:) = u(:,1);
        S(f,:) = v(:,1);
        dec(f,:) = diag(s);
    end

    numImgColors = params.numImgColors;
    [perm,colors] = kmeans(C,numImgColors,'start','sample','maxiter',1000,'replicates',10,'EmptyAction','singleton');
    [ordsort,popo]=sort(perm);
    
    for f=1:size(W,1)
        chunk = (C(perm(f),:)')*dec(f,1)*(S(f,:));
        Wapprox(f,:,:,:) = reshape(chunk,1,size(W,2),size(W,3),size(W,4));
    end
    
    Wapprox = permute(Wapprox, [1, 3, 4, 2]);
    Wmono = reshape(S,size(W,1),size(W,3),size(W,4));
    
    args = struct('colors', colors, 'Wmono', Wmono, 'perm', perm);
    args.cuda_template = strcat(morf_path, 'cuda/src/filter_acts_mono_template.cuh');
    args.cuda_true = strcat(morf_path, 'cuda/src/filter_acts_mono.cuh'); 
    args.compile_path = strcat(morf_path, 'cuda/');
    args.X = X;
    args.W = W;
    args.Wmono = Wmono;
    args.Wapprox = Wapprox;
    args.colors = colors;
    args.perm = perm;
    
    
    
end