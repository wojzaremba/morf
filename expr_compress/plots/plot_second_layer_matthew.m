global plan root_path
init();
type = 'matthew';
load_imagenet_model(type);

W = plan.layer{5}.cpu.vars.W;

args.iclust = 12;
args.oclust = 32;
args.k = 12;
[Wapprox, F, C, X, Y, assignment] = bisubspace_lowrank_approx(double(W), args);

rast = 1;
chunk = W( :, :, :, assignment{rast}.Ii);
chunk = chunk(:, :);
ydata = tsne(chunk, [], 3);
scatter3(ydata(:, 1), ydata(:, 2), ydata(:, 3), 50, 'filled');

% figure1 = figure('Position', [0, 0, 1000, 1000]);
% axes1 = axes('Parent',figure1);
% view(axes1,[-120.5 48]);
% grid(axes1,'on');
% hold(axes1,'all');

WW = W(:, :);
[u, s, v] = svd(WW);

%saveas(figure1, sprintf('expr_compress/paper/img/secnd_layer_%s_3d', type), 'epsc');