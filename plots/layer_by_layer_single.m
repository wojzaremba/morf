clear all
a = load('/Users/wojto/Dropbox/2013/compress/mat_files/eval_per_batch_fft', 'times_fft');
b = load('/Users/wojto/Dropbox/2013/compress/mat_files/eval_per_batch', 'timess');
times_fft = a.times_fft;
timess = b.timess;
labels = {};
json = ParseJSON('plans/imagenet.txt');
relevant = [];
for i = 2:(length(json) - 1)
    type = json{i}.type;
    if (strcmp(type, 'NNShared'))
        type = 'Conv';
        relevant = [relevant; i];
    else
        times_fft(i, :) = 0;
        if (strcmp(type, 'NN'))
            type = 'FC';
        elseif (strcmp(type, 'MaxPooling'))
            type = 'MaxPool';
        end
    end
    labels{end + 1} = type;
end
hFig = figure(1);
set(hFig, 'Position', [0 0 1200 400]);
Y1 = squeeze(mean(timess(2:(end - 1), :, 1), 2));
Y2 = squeeze(mean(times_fft(2:(end - 1), :, 1), 2));
bar(1:14', [Y1, Y2]);
hold on
errorbar((1:14)' - 0.13, Y1, squeeze(std(timess(2:(end - 1), :, 1), [], 2)), '.')
errorbar(relevant' - 1 + 0.13, Y2(relevant - 1), squeeze(std(times_fft(relevant, :, 1), [], 2)), '.')
set(gca, 'XTickLabel', labels, 'FontSize', 12);
ylabel('CPU time for a single image FP (s)', 'FontSize', 18);
set(gca, 'Ticklength', [0 0])
ylim([0 max(Y1) * 1.05])
legend('Convolution in spacial domain', 'Convolution in Fourier domain');

path = '/Users/wojto/Dropbox/2013/compress/img/eval_per_layer_per_batch_1_batch_size.png';
img = imread(path);
f = find(mean(mean(img, 2), 3) ~= 255);
ox = f(1);
ex = f(end);
f = find(mean(mean(img, 1), 3) ~= 255);
oy = f(1);
ey = f(end);
imwrite(img(ox:ex, oy:ey, :), path);