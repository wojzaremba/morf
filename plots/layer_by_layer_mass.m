clear all
b = load('/Users/wojto/Dropbox/2013/compress/mat_files/eval_per_batch_GPU', 'timess');
% XXX : It is little bit lame that this graph uses results from two very
% different implementations. Resolve it.

% Results by Micheal
% 2nd convolution (FFT)
% 69.849
% 69.8509
% 70.488
% 76.6501
% 73.2911
% 69.854
% 69.859
% 77.61
% 
% NOT FFT
% 99.16
% 105.52
% 116.70
% 106.67
% 106.64
% 
% 3rd convolution (FFT)
% 78.274
% 75.597
% 79.5591
% 76.2191
% 77.7209
% 
% NOT FFT
% 34.23
% 68.26
% 37.15
% 37.18
% 37.18
% 
% 4th convolution (FFT)
% 107.722
% 116.568
% 108.281
% 115.223
% 119.262
% 
% NOT FFT
% 50.301
% 55.735
% 55.913
% 55.9391
% 60.924

times_fft = zeros(size(b.timess));
times_fft(5, :, 8) = [69.8509, 70.488, 76.6501, 73.2911, 69.854] / 1000;

timess = 128 * b.timess;
labels = {};
json = ParseJSON('plans/imagenet.txt');
relevant = [];
for i = 2:(length(json) - 1)
    type = json{i}.type;
    if (strcmp(type, 'Conv'))
        relevant = [relevant; i];
    else
        times_fft(i, :) = 0;
        if (strcmp(type, 'MaxPooling'))
            type = 'MaxPool';
        end
    end
    labels{end + 1} = type;
end
hFig = figure(1);
set(hFig, 'Position', [0 0 1200 400]);
Y1 = squeeze(mean(timess(2:(end - 1), :, 8), 2));
Y2 = squeeze(mean(times_fft(2:(end - 1), :, 8), 2));
bar(1:13', [Y1, Y2]);
hold on
errorbar((1:13)' - 0.13, Y1, squeeze(std(timess(2:(end - 1), :, 8), [], 2)), '.')
errorbar(relevant - 1 + 0.13, Y2(relevant - 1), squeeze(std(times_fft(relevant, :, 8), [], 2)), '.')

set(gca, 'XTickLabel', labels, 'FontSize', 12);
ylabel('GPU time for a single image FP (s)', 'FontSize', 18);
set(gca, 'Ticklength', [0 0])
ylim([0 max(Y1) * 1.1])
legend('Convolution in spacial domain', 'Convolution in Fourier domain');

path = '/Users/wojto/Dropbox/2013/compress/img/eval_per_layer_per_batch_GPU_128_batch_size.png';
img = imread(path);
f = find(mean(mean(img, 2), 3) ~= 255);
ox = f(1);
ex = f(end);
f = find(mean(mean(img, 1), 3) ~= 255);
oy = f(1);
ey = f(end);
imwrite(img(ox:ex, oy:ey, :), path);