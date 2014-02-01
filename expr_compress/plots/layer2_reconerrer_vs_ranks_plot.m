figure1 = figure('Position', [0, 0, 700, 500]);

load layer2reconerror_vs_numweights_matthew_iclust=32_oclust=16_kmeans
plot(ranks, recon_error, 'linewidth', 2);
hold on;

load layer2reconerror_vs_numweights_matthew_iclust=32_oclust=4_kmeans
plot(ranks, recon_error, 'm-.', 'linewidth', 2);
hold on;

load layer2reconerror_vs_numweights_matthew_iclust=48_oclust=4_kmeans
plot(ranks, recon_error, 'r--', 'linewidth', 2);

load layer2reconerror_vs_numweights_matthew_iclust=48=oclust=1_kmeans
plot(ranks, recon_error, 'g:', 'linewidth', 2);

xlabel('Rank of approximation for each input output cluster pair', 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
ylabel('Reconstruction error', 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
legend1 = legend('G = 32; H = 16', ...
                 'G = 32; H = 4', ...
                 'G = 48; H = 4', ...
                 'G = 48; H = 1');
 
set(legend1,'Position',[0.640714285714284 0.746 0.217857142857144 0.126]);

saveas(figure1, 'expr_compress/paper/img/layer2reconerror_vs_ranks', 'epsc');
