function collage=collage_imnet(aux)

S=size(aux);

N=16;
M=floor(S(1)/N);

marg=2;



collage = zeros( (S(3)+marg)*N, (S(4)+marg)*M, 3);

for rast=1:S(1)
rn = mod(rast-1,N)+1;
rm = floor((rast-1)/N)+1;
collage( (S(3)+marg)*(rn-1)+1:(S(3)+marg)*(rn-1)+S(3), (S(4)+marg)*(rm-1)+1:(S(4)+marg)*(rm-1)+S(4),:)=colorescale(permute(squeeze(aux(rast,:,:,:)),[2 3 1]));

end
figure
imagesc(collage)

