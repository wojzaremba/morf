load('../data/imagenet/test.mat')
for i = 1:length(data)
    X = zeros(3, 227, 227);
    X(:, 2:225, 2:225) = data{i}.X;
    data{i}.X = X;
end
save('../data/imagenet/test.mat', 'data')
