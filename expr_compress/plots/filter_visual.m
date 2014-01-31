clc;
clear all;
global plan;
sizes = [7, 11];
names = {'matthew', 'alex'};
for k = 1:length(names)
    name = names{k};
    s = sizes(k);
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
    Q = repmat(reshape(Q, [1, 1, 1, 3]), [96, s, s, 1]);
    W = W - Q; 
    Wapprox = Wapprox - Q; 
    Q = permute([W; Wapprox], [4, 1, 2, 3]);
    Q = Q(:, :);
    Q = max(Q, [], 2) + 1e-4;
    Q = repmat(reshape(Q, [1, 1, 1, 3]), [96, s, s, 1]);
    W = W ./ Q;
    Wapprox = Wapprox ./ Q;

    img = ones((s + 1) * 12 - 1, 2 * 8 * (s + 6) - 10, 3);
    for i = 1:12
        for j = 1:8
            idx = (j - 1) * 12 + i;
            img(((i - 1) * (s + 1) + 1):((i - 1) * (s + 1)  + s), ((j - 1) * 2 * (s + 6) + 1):((j - 1) * 2 * (s + 6) + s), :) = squeeze(W(idx, :, :, :));
            img(((i - 1) * (s + 1) + 1):((i - 1) * (s + 1)  + s), ((j - 1) * 2 * (s + 6) + 3 + s):((j - 1) * 2 * (s + 6) + 2 + 2 * s), :) = squeeze(Wapprox(idx, :, :, :));
        end
    end
    imagesc(img);
    imwrite(img, sprintf('expr_compress/paper/img/denoising_%s.png', name));
end