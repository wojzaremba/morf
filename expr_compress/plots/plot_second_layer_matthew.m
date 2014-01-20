global plan root_path
init();
type = 'matthew';
load_imagenet_model(type);

W = plan.layer{5}.cpu.vars.W;


figure1 = figure('Position', [0, 0, 1000, 1000]);
axes1 = axes('Parent',figure1);
view(axes1,[-120.5 48]);
grid(axes1,'on');
hold(axes1,'all');

%saveas(figure1, sprintf('expr_compress/paper/img/secnd_layer_%s_3d', type), 'epsc');