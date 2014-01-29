clear all;
addpath(genpath('..'));
global plan
json = ParseJSON('plans/mnist_conv.txt');
bs = 100;
json{1}.batch_size = bs;
Plan(json, 'trained/mnist_conv');
plan.training = 0;
plan.input.step = 1;
plan.input.GetImage(0);

plan.lr = 0;
Xorg = plan.input.cpu.vars.out;
Yorg = plan.input.cpu.vars.Y;

oY = zeros(size(plan.input.cpu.vars.Y));
oY(:, 2:end) = plan.input.cpu.vars.Y(:, 1:(end - 1));
oY(:, 1) = plan.input.cpu.vars.Y(:, end);
plan.input.cpu.vars.Y = oY;
for iter = 1:150
    ForwardPass(plan.input);
    score = 0;
    fprintf('Still correctly classified samples : ');
    for i = 1:bs
        [~, idx] = max(plan.classifier.cpu.vars.pred(i, :));
        score = score + Yorg(i, idx);
        if (Yorg(i, idx)) fprintf('%d, ', i);end;        
    end
    fprintf('\nscore = %f, iter = %d\n', score, iter);
    dX = zeros(json{1}.batch_size, 28 * 28, 10);
    for i = length(plan.layer):-1:2
        layer = plan.layer{i};
        layer.BP();
        plan.layer{i - 1}.cpu.dvars.out = layer.cpu.dvars.X;
    end
    plan.input.cpu.vars.out(:, :) = plan.input.cpu.vars.out(:, :) - 0.01 * plan.layer{2}.cpu.dvars.X(:, :);
    plan.input.cpu.vars.out = min(max(plan.input.cpu.vars.out, 0), 1);
end
colormap(gray)
img = zeros(10 * 28, 20 * 28);
for i = 1:10
    for j = 1:10
        img(((i - 1) * 28 + 1):(i * 28), ((2 * j - 1) * 28 + 1):(2 * j * 28)) = plan.input.cpu.vars.out((i - 1) * 10 + j, :, :);
        img(((i - 1) * 28 + 1):(i * 28), ((2 * j - 2) * 28 + 1):((2 * j - 1) * 28)) = Xorg((i - 1) * 10 + j, :, :);
    end
end
imagesc(img);