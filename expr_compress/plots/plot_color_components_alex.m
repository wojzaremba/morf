global plan root_path
init();
type = 'alex';
load_imagenet_model(type);

W = plan.layer{2}.cpu.vars.W;
K = 11;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Plot original weights
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure1 = figure('Position', [0, 0, 1000, 1000]);
axes1 = axes('Parent',figure1);
view(axes1,[-165.5 36]);
grid(axes1,'on');
hold(axes1,'all');

to_plot = 1:1:96;
WW = permute(W, [4, 2, 3, 1]);
WW = WW(:, :, :, to_plot);
WW = WW(:, :)';

colors = repmat(1:length(to_plot), [K, K, 1]);

scatter3(WW(:,1), WW(:,2), WW(:,3), 10, colors(:), 'filled');
hold on

set(gca,'XtickLabel',[],'YtickLabel',[],'ZtickLabel',[]);
axis(0.4 * [-1 1 -1 1 -1 1]);
saveas(figure1, sprintf('expr_compress/paper/img/color_components_%s_3d', type), 'epsc');