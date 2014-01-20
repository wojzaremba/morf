global plan root_path
init();
type = 'alex';
load_imagenet_model(type);

W = plan.layer{2}.cpu.vars.W;
K = 11;

figure1 = figure('Position', [0, 0, 1000, 1000]);
axes1 = axes('Parent',figure1);
view(axes1,[-120.5 48]);
grid(axes1,'on');
hold(axes1,'all');

to_plot = 1:1:86;
WW = permute(W, [4, 2, 3, 1]);
WW = WW(:, :, :, to_plot);
WW = WW(:, :)';

scatter3(WW(:,1), WW(:,2), WW(:,3), 50, colors{q});
hold on
%     plane = find(order == q);
%     WW = permute(W, [4, 2, 3, 1]);
%     WW = WW(:, plane);
%     [u, s, v] = svd(WW);
%     u = u(:, 1:2);
%     x = zeros(10, 10);
%     y = zeros(size(x));
%     z = zeros(size(x));
%     c = ones(size(x));
%     for i = 1:size(x, 1)
%         for j = 1:size(x,2)
%             idx1 = i - (size(x, 1) / 2);
%             idx2 = j - (size(x, 2) / 2);
%             x(i, j) = idx1 * u(1, 1) + idx2 * u(1, 2);
%             y(i, j) = idx1 * u(2, 1) + idx2 * u(2, 2);
%             z(i, j) = idx1 * u(3, 1) + idx2 * u(3, 2);
%         end
%     end
%     scale = 1.70 * max(abs([x(:); y(:); z(:)]));
%     x = x / scale;
%     y = y / scale;
%     z = z / scale;
%     surf(x,y,z,c);
%     colormap hsv
%     alpha(.1)    

set(gca,'XtickLabel',[],'YtickLabel',[],'ZtickLabel',[]);
axis(0.4 * [-1 1 -1 1 -1 1]);
saveas(figure1, sprintf('expr_compress/paper/img/color_components_%s_3d', type), 'epsc');