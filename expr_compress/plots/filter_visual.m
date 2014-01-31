clc;
clear all;
global plan;
sizes = [7, 11];
names = {'matthew', 'alex'};
for k = 1:length(names)
    name = names{k};
    json_old = ParseJSON(sprintf('plans/imagenet_%s.txt', name));
    json = {};
    json{1} = json_old{1};
    json{2} = json_old{2};

    json{1}.batch_size = 1;
    Plan(json, sprintf('~/imagenet_data/imagenet_%s', name), 0, 'single');

    W_ = plan.layer{2}.cpu.vars.W;

    [Wapprox_, Wmono, colors, perm] = monochromatic_approx(double(W_), struct('num_colors', 12, 'even', 1, 'start', 'sample'));

    W = W_;
    Wapprox = Wapprox_;

    Q = permute([W; Wapprox], [4, 1, 2, 3]);
    Q = Q(:, :);
    Q = min(Q, [], 2) - 1e-4;
    Q = repmat(reshape(Q, [1, 1, 1, 3]), [96, sizes(k), sizes(k), 1]);
    W = W - Q; 
    Wapprox = Wapprox - Q; 
    Q = permute([W; Wapprox], [4, 1, 2, 3]);
    Q = Q(:, :);
    Q = max(Q, [], 2) + 1e-4;
    Q = repmat(reshape(Q, [1, 1, 1, 3]), [96, sizes(k), sizes(k), 1]);
    W = W ./ Q;
    Wapprox = Wapprox ./ Q;

    img = ones(sizes(k) * 12, 2 * 10 * sizes(k), 3);
    for i = 1:12
        for j = 1:8
            idx = (j - 1) * 8 + i;
            img(((i - 1) * 8 + 2):((i - 1) * 8 + 1 + sizes(k)), ((j - 1) * 2 * 10 + 3):((j - 1) * 2 * 10 + 2 + sizes(k)), :) = squeeze(W(idx, :, :, :));
            img(((i - 1) * 8 + 2):((i - 1) * 8 + 1 + sizes(k)), ((j - 1) * 2 * 10 + 4 + sizes(k)):((j - 1) * 2 * 10 + 3 + 2 * sizes(k)), :) = squeeze(Wapprox(idx, :, :, :));
        end
    end
    imagesc(img);
    imwrite(img, sprintf('expr_compress/paper/img/denoising_%s.png', name));
end