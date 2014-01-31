num_colors = [1, 2, 4, 6, 8, 12, 16, 24, 32, 64, 96];

figure1 = figure('Position', [0, 0, 700, 500]);

% Matt
C = 3;
F = 96;
X = 7;
N = 224;
M = 110;

approx_ops = [];
for i = 1:length(num_colors)
   CC = num_colors(i);
   no = CC * C^2 * N * N + X * X * F * M * M;
   approx_ops = [approx_ops, no]; 
end

orig_ops = C * X * X * F * M * M;
plot(num_colors, orig_ops ./ approx_ops, 'linewidth', 2);
hold on;

% Alex
C = 3;
F = 96;
X = 11;
N = 224;
M = 55;

approx_ops = [];
for i = 1:length(num_colors)
   CC = num_colors(i);
   no = CC * C^2 * N * N + X * X * F * M * M;
   approx_ops = [approx_ops, no]; 
end

orig_ops = C * X * X * F * M * M;
plot(num_colors, orig_ops ./ approx_ops, 'r--', 'linewidth', 2);


xlabel('Number of colors used in monochromatic approximation', 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
ylabel('Relative decrease in number of operations', 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
%title('Relative decrease in number of operations for various monochromatic operations applied to MattNet');
legend1 = legend('MattNet', 'AlexNet');

set(legend1,'Position',[0.695714285714286 0.746 0.175714285714286 0.126]);

saveas(figure1, 'expr_compress/paper/img/monochromatic_numcolors_vs_numops', 'epsc');