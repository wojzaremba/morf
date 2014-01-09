clear all
q = load('../data/imagenet/imagenet-test-batch.mat');
data = cell(128, 1);
for i = 1:128
    data{i}.X = reshape(q.data(:, i)', 224, 224, 3);
    data{i}.Y = zeros(1000, 1);
    data{i}.Y(q.labels(i) + 1) = 1;
end
save('../data/imagenet/test', 'data');