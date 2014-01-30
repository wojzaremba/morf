f_size = 255;
f = dir('~/n01440764/');
X = zeros(length(f) - 2, f_size, f_size, 3);
for i = 3:length(f)
    fprintf('%d\n', i);
    name = ['~/n01440764/', f(i).name];
    A = imread(name);
    X(i - 2, :, :, :) = ExtractCentral(A, f_size);
end

X = X - repmat(mean(X, 1), [size(X, 1), 1, 1, 1]);

data = {};
for i = 1:size(X, 1);
    data{i}.X = permute(squeeze(X(i, :, :, :)), [3, 1, 2]);
    tmp = zeros(1000, 1);
    data{i}.Y = tmp;
end
save('../data/imagenet/train', 'data')
nr = 1;
X = [reshape(data{3}.X, [1, 3, 255, 255]); reshape(data{5}.X, [1, 3, 255, 255])];
X = X(:, :, 14:240, 14:240);
% imagesc(min(max((permute(squeeze(oXT(2, :, :, :) - oX(2, :, :, :)), [2, 3, 1]) + 200) / 400, 0), 1))

imagesc(min(max((permute(squeeze(A(1, :, :, :)), [2, 3, 1]) + 200) / 400, 0), 1))