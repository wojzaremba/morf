global plan root_path
init();
type = 'alex';
load_imagenet_model(type);

W = plan.layer{2}.cpu.vars.W;
K = 11;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Plot monochromatic approximated weights
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure1 = figure('Position', [0, 0, 1000, 1000]);
axes1 = axes('Parent',figure1);
view(axes1,[14.5 34]);
grid(axes1,'on');
hold(axes1,'all');

args.num_colors = 12;
args.even = 0;
[Wapprox, Wmono, colors, perm] = monochromatic_approx(W, args);
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

scatter3(WW(:,1), WW(:,2), WW(:,3), 15, colors(:), 'filled');
hold on

set(gca,'XtickLabel',[],'YtickLabel',[],'ZtickLabel',[]);
axis(0.4 * [-1 1 -1 1 -1 1]);
xlabel('R');
ylabel('G');
zlabel('B');
saveas(figure1, sprintf('expr_compress/paper/img/RGB_components_monoapprox%d_%s_3d', args.num_colors, type), 'epsc');