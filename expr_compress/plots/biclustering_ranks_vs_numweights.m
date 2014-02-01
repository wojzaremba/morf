ranks = [4, 6, 8, 12, 16, 24, 32, 64];

figure1 = figure('Position', [0, 0, 700, 500]);

% Matt
C = 96;
F = 256;
X = 5;
N = 224;
M = 110;

orig_weights = C * X * X * F;

Gs = [32, 32, 32, 48, 48, 48];
Hs = [16, 4, 1, 16, 4, 1]
col = [1, 0, 0; ...
       0, 1, 0; ...
       0, 0, 1;
       1, 1, 0; ...
       0, 1, 1; ...
       1, 0, 1;
       ];
for j = 1:length(Gs)
    G = Gs(j);
    H = Hs(j);
    approx_weights = [];
    for i = 1:length(ranks)
       K = ranks(i);      
       nw =  G * H * K * ((C/G) + X*X + (F/H));
       approx_weights = [approx_weights, nw]; 
    end

    plot(ranks, orig_weights ./ approx_weights, 'linewidth', 2, 'Color', col(j, :));
    hold on;
end
title('MattNet', 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
xlabel('Rank of approximation for each input output cluster pair', 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
ylabel('Relative decrease in number of weights', 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
legend1 = legend('G = 32; H = 16', ...
                 'G = 32; H = 4', ...
                 'G = 32; H = 1', ...
                 'G = 48; H = 16', ...
                 'G = 48; H = 4', ...
                 'G = 48; H = 1');

set(legend1,'Position',[0.695714285714286 0.746 0.175714285714286 0.126]);

saveas(figure1, 'expr_compress/paper/img/biclustering_rank_vs_numweights_matt', 'epsc');


figure1 = figure('Position', [0, 0, 700, 500]);

% Alex
C = 96;
F = 256;
X = 5;
N = 27;
M = 27;

Gs = [32, 32, 32, 48, 48, 48];
Hs = [16, 4, 1, 16, 4, 1]
col = [1, 0, 0; ...
       0, 1, 0; ...
       0, 0, 1;
       1, 1, 0; ...
       0, 1, 1; ...
       1, 0, 1;
       ];
for j = 1:length(Gs)
    G = Gs(j);
    H = Hs(j);
    approx_weights = [];
    for i = 1:length(ranks)
       K = ranks(i);      
       nw =  G * H * K * ((C/G) + X*X + (F/H));
       approx_weights = [approx_weights, nw]; 
    end

    plot(ranks, orig_weights ./ approx_weights, 'linewidth', 2, 'Color', col(j, :));
    hold on;
end

title('AlexNet', 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
xlabel('Rank of approximation for each input output cluster pair', 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
ylabel('Relative decrease in number of weights', 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
legend1 = legend('G = 32; H = 16', ...
                 'G = 32; H = 4', ...
                 'G = 32; H = 1', ...
                 'G = 48; H = 16', ...
                 'G = 48; H = 4', ...
                 'G = 48; H = 1');

set(legend1,'Position',[0.695714285714286 0.746 0.175714285714286 0.126]);

saveas(figure1, 'expr_compress/paper/img/biclustering_rank_vs_numweights_alex', 'epsc');