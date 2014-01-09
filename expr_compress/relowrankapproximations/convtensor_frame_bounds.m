function spectrum=convtensor_frame_bounds(W)

W = permute(W,[1 4 2 3]);
S=size(W);

R0=8;
%Ws=zeros(S(1),S(2),S(3)*S(4));
Ws=zeros(S(1),S(2),R0^2);

for i=1:S(1)
    for j=1:S(2)
    aux = fft2(squeeze(W(i,j,:,:)),R0,R0);
    Ws(i,j,:)=aux(:);
    end
end
for i=1:size(Ws,3)
tmp = squeeze(Ws(:,:,i));
la(:,i)=svd(tmp,0);
end

spectrum=sort(la(:),'descend');
%out(1)=min(la(:));
%out(2)=max(la(:));

