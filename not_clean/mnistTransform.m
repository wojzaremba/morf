% load('../data/imagenet/train.mat');
for i = 1:length(data)
    data{i}.X = permute(data{i}.X, [2, 3, 1]);
end
% save('../data/imagenet/train.mat', 'data');