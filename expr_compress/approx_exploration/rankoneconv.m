function [C, H, V, F, Xout] = rankoneconv(X, K) % XXX : name changed to temporarily fix naming problem
%this function approximates the 4-D convolutional tensor X by k rank-1 tensors.
%joan bruna 2013 Courant Institute

    R=X;
    Xout=0*X;
    S=size(X);
    C=zeros(S(1),K);
    H=zeros(S(2),K);
    V=zeros(S(3),K);
    F=zeros(S(4),K);

    niters=64;

    inputn=norm(R(:));
    for k=1:K
        [C(:,k), H(:,k), V(:,k), F(:,k),aux]=rankonetens(R,niters);
        R=R-aux;
        Xout=Xout+aux;
        if mod(k,16)==0
            fprintf('it %d residual norm is %f \n', k, norm(R(:))/inputn )
        end

    end


end


function [C, H, V, F, out]=rankonetens(X,niters)

    S=size(X);
    %init
    tmp=permute(X,[1 2 3 4]);
    tmp=tmp(:,:);
    C = mean(tmp,2);
    tmp=permute(X,[2 1 3 4]);
    tmp=tmp(:,:);
    H = mean(tmp,2);
    tmp=permute(X,[3 1 2 4]);
    tmp=tmp(:,:);
    V = mean(tmp,2);
    tmp=permute(X,[4 1 3 2]);
    tmp=tmp(:,:);
    F = mean(tmp,2);

    for n=1:niters

        %C
        tmp=permute(X,[1 2 3 4]);
        tmp=tmp(:,:);
        aux=tensorize(H,V,F);
        C = tmp * aux(:);

        %H
        tmp=permute(X,[2 1 3 4]);
        tmp=tmp(:,:);
        aux=tensorize(C,V,F);
        H = tmp * aux(:);

        %V
        tmp=permute(X,[3 1 2 4]);
        tmp=tmp(:,:);
        aux=tensorize(C,H,F);
        V = tmp * aux(:);

        %F
        tmp=permute(X,[4 1 2 3]);
        tmp=tmp(:,:);
        aux=tensorize(C,H,V);
        F = tmp * aux(:);

        C=C/norm(C);
        H=H/norm(H);
        V=V/norm(V);
        F=F/norm(F);

    end

    tmp = reshape(X,S(1)*S(2)*S(3),S(4));
    tmp = tmp * F;
    tmp = reshape(tmp, S(1)*S(2), S(3));
    tmp = tmp * V;
    tmp = reshape(tmp, S(1), S(2));
    tmp = tmp * H;
    lambda = sum(tmp.*C);
    %end

    out=tensorize(lambda*C, H, V, F);

end



