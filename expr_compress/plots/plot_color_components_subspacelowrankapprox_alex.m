global plan root_path
init();
type = 'alex';
load_imagenet_model(type);

W = plan.layer{2}.cpu.vars.W;
K = 11;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Plot subspace clustered low rank weights
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
args.num_colors = 3;
args.terms_per_element = 1;
[Wapprox, F, C, X, Y] = subspace_lowrank_approx(double(W), args);
figure1 = figure('Position', [0, 0, 1000, 1000]);
axes1 = axes('Parent',figure1);
view(axes1,[14.5 34]);
grid(axes1,'on');
hold(axes1,'all');

to_plot = 1:1:96;
WW = permute(Wapprox, [4, 2, 3, 1]);
WW = WW(:, :, :, to_plot);
WW = WW(:, :)';

colors = repmat(1:length(to_plot), [K, K, 1]);

scatter3(WW(:,1), WW(:,2), WW(:,3), 20, colors(:), 'filled');
hold on

set(gca,'XtickLabel',[],'YtickLabel',[],'ZtickLabel',[]);
axis(0.4 * [-1 1 -1 1 -1 1]);
saveas(figure1, sprintf('expr_compress/paper/img/color_components_subspacelowrankapprox%d-%d_%s_3d', args.num_colors, args.terms_per_element, type), 'epsc');




