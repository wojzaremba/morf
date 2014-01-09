clear all
load('/Users/wojto/Dropbox/2013/compress/mat_files/eval_per_batch_GPU', 'timess');
figure(1)
errorbar(0:7, 1000 * squeeze((mean(sum(timess, 1), 2))), 1000 * squeeze((std(sum(timess, 1), 0, 2))));
set(gca, 'XTickLabel',[]);

% set(gca, 'FontSize',16);
% set(gca,'YTick',[0:9]);
set(gca,'YTick',[0:10:220]);
ylim([0, 220]);
xlim([0, 7]);

xt = get(gca, 'XTick');
yl = get(gca, 'YLim');
str = cellstr( num2str(xt(:),'2^{%d}') );      %# format x-ticks as 2^{xx}
hTxt = text(xt, yl(ones(size(xt))), str, ...   %# create text at same locations
    'Interpreter','tex', ...                   %# specify tex interpreter
    'VerticalAlignment','top', ...             %# v-align to be underneath
    'HorizontalAlignment','center'); 

xlabel('Batch size', 'FontSize', 18);
ylabel('GPU time per image (ms)', 'FontSize', 18);
xlabh = get(gca,'XLabel');
set(xlabh,'Position',get(xlabh,'Position') - [0 .1 0]);

text(0.15, 210, 'Real time applications',...
	'VerticalAlignment','middle',...
	'HorizontalAlignment','left',...
	'FontSize', 16, 'FontWeight', 'bold');


text(4, 25, 'Mass scale annotation',...
	'VerticalAlignment','middle',...
	'HorizontalAlignment','left',...
	'FontSize', 16, 'FontWeight', 'bold');

saveas(gca, '/Users/wojto/Dropbox/2013/compress/img/eval_per_batch_GPU.png');