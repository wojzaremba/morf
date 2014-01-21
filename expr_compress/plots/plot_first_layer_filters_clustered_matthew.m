global plan root_path
init();
type = 'matthew';
load_imagenet_model(type);
load('Grps_subspace_clustering_two_planes.mat');

W = plan.layer{2}.cpu.vars.W;
K = 7;
order = Grps(:, 2);

plane2 = reshape(order == 2, [7, 7, 96]);
p2 = permute(plane2, [3, 1, 2]);
order = ones(96, 1);
order(find(sum(p2(:, :), 2) > 25)) = 2;

WW = zeros(size(W));
WW(1:sum(order == 1), :, :, :) = W(order == 1, :, :, :);
WW(sum(order == 1) + 1 : end, :, :, :) = W(order == 2, :, :, :);
WW = permute(WW, [1, 4, 2, 3]);

figure1 = figure('Position', [0, 0, 650, 1000]);


S=size(WW);

N=12;
M=floor(S(1)/N);

marg=1;



collage = ones( (S(3)+2*marg)*N, (S(4)+2*marg)*M, 3);
rgb = [0.5, 0.1, 0.1; 0.1, 0.1, 0.5];
for c = 1 : 3
    collage(:, :, c) = rgb(2, c);
    collage(:, 1:(S(4)+2*marg)*4, c) = rgb(1, c);
    collage(1:(S(3)+2*marg)*8, 1:(S(4)+2*marg)*5, c) = rgb(1, c);
end

for rast=1:S(1)
    rn = mod(rast-1,N)+1;
    rm = floor((rast-1)/N)+1;
    collage( (S(3) + 2*marg)*(rn-1)+1+marg:(S(3) + 2*marg)*(rn-1)+S(3)+marg,  (S(4) + 2*marg)*(rm-1)+1+marg:(S(4) + 2*marg)*(rm-1)+S(4)+marg,:)=colorescale(permute(squeeze(WW(rast,:,:,:)),[2 3 1]));
end
imagesc(collage)

saveas(figure1, sprintf('expr_compress/paper/img/first_layer_filters_clustered_%s', type), 'epsc');