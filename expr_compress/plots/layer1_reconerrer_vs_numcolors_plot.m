figure1 = figure('Position', [0, 0, 700, 500]);

load layer1reconerror_vs_numweights_matthew_mykmeans.mat
plot(num_colors, recon_error, 'linewidth', 2);

xlabel('Number of colors used in monochromatic approximation', 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');
ylabel('Reconstruction error', 'FontSize', 15, 'FontName', 'TimesNewRoman', 'FontWeight', 'bold');

saveas(figure1, 'expr_compress/paper/img/layer1reconerror_vs_numcolors', 'epsc');
