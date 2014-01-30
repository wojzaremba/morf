function understanding_svd
    load('mats/mnist_800_dropout.mat');
    W{1} = plan.layer{2}.cpu.vars.W;
    load('mats/mnist_800.mat');
    W{2} = plan.layer{2}.cpu.vars.W;

    scale = 0.40;
    offset = 0.22;
    for k = 1:2
        img{k} = ones(30, 30 * 10 + 2) * scale - offset;
        bad{k} = ones(30, 30 * 10 + 2) * scale - offset;
        imgorig{k} = zeros(28, 28 * 10);
        [U, S, V] = svd(W{k});
        for i = 1:10
            img{k}(2:29, ((i - 1) * 30 + 3) : ((i - 1) * 30 + 30)) = reshape(U(:, i), 28, 28);
            imgorig{k}(:, ((i - 1) * 28 + 1) : (i * 28)) = reshape(U(:, i), 28, 28);
            bad{k}(2:29, ((i - 1) * 30 + 3) : ((i - 1) * 30 + 30)) = reshape(W{k}(:, i), 28, 28);
        end
        img_{k} = (img{k} + offset) / scale;
        bad_{k} = (bad{k} + offset) / scale;
        img_{k} = max(min(img_{k}, 1), 0);
        bad_{k} = max(min(bad_{k}, 1), 0);
    end
    cmap = colormap(gray);
    cmap(end, :) = 0;    
    imwrite(convertimg([bad_{1}; bad_{2}], cmap), '../paper/img/filters_mnist.png');
    imwrite(convertimg([img_{1}; img_{2}], cmap), '../paper/img/filters_svd_mnist.png');
    [U, ~, ~] = svd(W{1});
    inspectedges('svd dropout', U, 10);
    [U, ~, ~] = svd(W{2});
    inspectedges('svd no dropout', U, 10);    
    inspectedges('original dropout', W{1}, 800);
    inspectedges('original no dropout', W{2}, 800);
end

function inspectedges(name, M, range)
    top = [];
    bottom = [];
    left = [];
    right = [];
    for i = 1 : range
        tmp = reshape(M(:, i), 28, 28);
        top = [top, norm(tmp(1, :))];
        bottom = [bottom, norm(tmp(end, :))];
        left = [left, norm(tmp(:, 1))];
        right = [right, norm(tmp(:, end))];
    end
    fprintf('name = %s\n', name);
    fprintf('\ttop norm = %f, std = %f\n', mean(top), std(top));
    fprintf('\tbottom norm = %f, std = %f\n', mean(bottom), std(bottom));
    fprintf('\tleft norm = %f, std = %f\n', mean(left), std(left));
    fprintf('\tright norm = %f, std = %f\n', mean(right), std(right));
    fprintf('\n');
end

function ret = convertimg(img, cmap)
    img_ = round(img * 63) + 1;
    ret = zeros(size(img_, 1), size(img_, 2), 3);
    for i = 1 : size(img_, 1)
        for j = 1 : size(img_, 2)
            ret(i, j, :) = cmap(img_(i, j), :);
        end
    end
end



% imagesc([img{1}; bad{1}; img{2}; bad{2}]);
% 
% norm(bad{1}(1, :))
% norm(bad{2}(1, :))

% 1. show that on boundary there is less norm for dropout (for every out of
% 10 examples)
% write down norm of derivatives.

% Compare with just element-wise pictures of filters.

% weight decay.
% try with smaller number of features.

% Run experiemnts on tons of networks.


% XXXXXXX
% 1. Write down this understanding.
% 3. Work on visualization of second layer.