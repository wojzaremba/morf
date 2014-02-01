figure1 = figure('Position', [0, 0, 700, 500]);

load layer1reconerror_vs_numweights_matthew_mykmeans.mat
plot(num_colors, test_error ./ 1024, 'linewidth', 2);
hold on;
plot(num_colors, 0.168 * ones(size(num_colors)), 'k');

xlabel('Number of colors used in monochromatic approximation', 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
ylabel('Test error', 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');

saveas(figure1, 'expr_compress/paper/img/layer1testerror_vs_numcolors', 'epsc');
