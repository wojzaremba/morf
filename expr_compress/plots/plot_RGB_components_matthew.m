global plan root_path
init();
type = 'matthew';
load_imagenet_model(type);
load('Grps_subspace_clustering_two_planes.mat');

W = plan.layer{2}.cpu.vars.W;
K = 7;

figure1 = figure('Position', [0, 0, 1000, 1000]);
axes1 = axes('Parent',figure1);
view(axes1,[-120.5 48]);
grid(axes1,'on');
hold(axes1,'all');

order = Grps(:, 2);
colors = {'rx', 'bo'};
for q = 1:2
    to_plot = find(order == q);
    WW = permute(W, [4, 2, 3, 1]);
    WW = WW(:, :);
    WW = WW(:, to_plot)';

    scatter3(WW(:,1), WW(:,2), WW(:,3), 50, colors{q});
    hold on
    plane = find(order == q);
    WW = permute(W, [4, 2, 3, 1]);
    WW = WW(:, plane);
    [u, s, v] = svd(WW);
    u = u(:, 1:2);
    x = zeros(10, 10);
    y = zeros(size(x));
    z = zeros(size(x));
    c = ones(size(x));
    for i = 1:size(x, 1)
        for j = 1:size(x,2)
            idx1 = i - (size(x, 1) / 2);
            idx2 = j - (size(x, 2) / 2);
            x(i, j) = idx1 * u(1, 1) + idx2 * u(1, 2);
            y(i, j) = idx1 * u(2, 1) + idx2 * u(2, 2);
            z(i, j) = idx1 * u(3, 1) + idx2 * u(3, 2);
        end
    end
    scale = 1.70 * max(abs([x(:); y(:); z(:)]));
    x = x / scale;
    y = y / scale;
    z = z / scale;
    surf(x,y,z,c);
    colormap hsv
    alpha(.1)    
end
set(gca,'XtickLabel',[],'YtickLabel',[],'ZtickLabel',[]);
axis(0.4 * [-1 1 -1 1 -1 1]);
xlabel('R');
ylabel('G');
zlabel('B');
saveas(figure1, sprintf('expr_compress/paper/img/RGB_components_%s_3d', type), 'epsc');

figure1 = figure('Position', [0, 0, 1000, 1000]);
axes1 = axes('Parent',figure1);
view(axes1,[-115.5 38]);
grid(axes1,'on');
hold(axes1,'all');

order = Grps(:, 2);
colors = {'rx', 'bo'};
for q = 1:2
    to_plot = find(order == q);
    WW = permute(W, [4, 2, 3, 1]);
    WW = WW(:, :);
    WW = WW(:, to_plot)';

    scatter3(WW(:,1), WW(:,2), WW(:,3), 50, colors{q});
    hold on
    plane = find(order == q);
    WW = permute(W, [4, 2, 3, 1]);
    WW = WW(:, plane);
    [u, s, v] = svd(WW);
    u = u(:, 1:2);
    x = zeros(10, 10);
    y = zeros(size(x));
    z = zeros(size(x));
    c = ones(size(x));
    for i = 1:size(x, 1)
        for j = 1:size(x,2)
            idx1 = i - (size(x, 1) / 2);
            idx2 = j - (size(x, 2) / 2);
            x(i, j) = idx1 * u(1, 1) + idx2 * u(1, 2);
            y(i, j) = idx1 * u(2, 1) + idx2 * u(2, 2);
            z(i, j) = idx1 * u(3, 1) + idx2 * u(3, 2);
        end
    end
    scale = 1.70 * max(abs([x(:); y(:); z(:)]));
    x = x / scale;
    y = y / scale;
    z = z / scale;
    surf(x,y,z,c);
    colormap hsv
    alpha(.1)    
end
set(gca,'XtickLabel',[],'YtickLabel',[],'ZtickLabel',[]);
axis(0.4 * [-1 1 -1 1 -1 1]);
xlabel('R');
ylabel('G');
zlabel('B');
saveas(figure1, sprintf('expr_compress/paper/img/RGB_components_%s_3d-2', type), 'epsc');