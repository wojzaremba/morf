function [Wapprox, recon_error] =  monochromatic_approx(W, num_colors)
    % W dimensions : (Nout, X, Y, Nin)
    
    W = permute(W, [1, 4, 2, 3]);
    for f =1 : size(W, 1)
        [u,s,v] = svd(squeeze(W(f, :, :)), 0);
        C(f, :) = u(:, 1);
        S(f, :) = v(:, 1);
        dec(f, :) = diag(s);
    end

    [assignment, colors] = litekmeans(C', num_colors);
    colors = colors';
    
    % Reconstruct weights
    Wapprox = zeros(size(W));
    for f = 1 : size(W, 1)
        chunk = (colors(assignment(f), :)') * dec(f, 1) * (S(f, :));
        Wapprox(f, :, :, :) = reshape(chunk, [1, size(Wapprox, 2), size(Wapprox, 3), size(Wapprox, 4)]); %[96, 3, 11, 11]
    end  
    
    % Permute back to original dimension ordering
    Wapprox = permute(Wapprox, [1, 3, 4, 2]);
    W = permute(W, [1, 3, 4, 2]);
    
    recon_error = norm(W(:) - Wapprox(:)) / norm(W(:));
    
end
