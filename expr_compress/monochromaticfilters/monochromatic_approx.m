function [Wapprox, Wmono, colors, perm] = monochromatic_approx(W, num_colors)

    % Simple re-parametrization of first layer with monochromatic filters
    for f = 1 : size(W,1)
        [u,s,v]=svd(squeeze(W(f,:,:)),0);
        C(f,:)=u(:,1);
        S(f,:)=v(:,1);
        dec(f,:)=diag(s);
        chunk = u(:,1)*s(1,1)*v(:,1)';
        approx0(f,:,:,:)=reshape(chunk,1,size(W,2),size(W,3),size(W,4));
    end

    MAXiter = 1000; % Maximum iteration for KMeans Algorithm
    REPlic = 10; % Replication for KMeans Algorithm
    [assignment,colors] = litekmeans(C',num_colors);

    colors = colors';
    Wapprox = zeros(size(W));
    for f=1:size(W,1)
        chunk = (colors(assignment(f),:)')*dec(f,1)*(S(f,:));
        Wapprox(f,:,:,:)=reshape(chunk,1,size(W,2),size(W,3),size(W,4));
    end 
    
    Wapprox = permute(Wapprox, [1, 3, 4, 2]);
    Wmono = reshape(bsxfun(@times, S, dec(:, 1)),size(W,1),size(W,3),size(W,4));
    
    assert(norm(squeeze(Wmono(1, :, :)) * colors(assignment(1), 1) + ...
                        squeeze(Wmono(1, :, :)) * colors(assignment(1), 2) + ...
                        squeeze(Wmono(1, :, :)) * colors(assignment(1), 3) - ...
                        (squeeze(Wapprox(1, :, :, 1)) + squeeze(Wapprox(1, :, :, 2)) + squeeze(Wapprox(1, :, :, 3)))) < 1e-10);
                    
    
    [~, perm] = sort(assignment);
    colors = colors';
    %Wmono = Wmono(perm, :, :);
    
    

end

