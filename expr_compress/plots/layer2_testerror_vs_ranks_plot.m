figure1 = figure('Position', [0, 0, 700, 500]);

load layer2reconerror_vs_numweights_matthew_iclust=32_oclust=16_kmeans
plot(ranks(1:end-2), test_error(1:end-2)/1024, 'linewidth', 2);
hold on;

load layer2reconerror_vs_numweights_matthew_iclust=32_oclust=4_kmeans
plot(ranks(1:end-2), test_error(1:end-2)/1024, 'm-.', 'linewidth', 2);
hold on;

load layer2reconerror_vs_numweights_matthew_iclust=48_oclust=4_kmeans
plot(ranks(1:end-2), test_error(1:end-2)/1024, 'r--', 'linewidth', 2);


load layer2reconerror_vs_numweights_matthew_iclust=48=oclust=1_kmeans
plot(ranks(1:end-2), test_error(1:end-2)/1024, 'g:', 'linewidth', 2);


plot(ranks(1:end-2), 0.168 * ones(size(ranks(1:end-2))), 'k');
xlabel('Rank of approximation for each input output cluster pair', 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
ylabel('Test error', 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');

legend1 = legend('G = 32; H = 16', ...
                 'G = 32; H = 4', ...
                 'G = 48; H = 4', ...
                 'G = 48; H = 1', ...
                 'Original weights');

set(legend1,'Position',[0.640714285714284 0.746 0.217857142857144 0.126]);

saveas(figure1, 'expr_compress/paper/img/layer2testerror_vs_ranks', 'epsc');
