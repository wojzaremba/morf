global plan
addpath(genpath('../'));
SmallImagenetModel(11);

examples = 5;
ox = 10; oy = 10;
out = zeros(256, floor((94 - 1) / ox) + 1, floor((94 - 1) / oy) + 1, examples);
t = zeros(2, size(out, 2), size(out, 3), examples);

for k = 1:examples
    [X, Y] = plan.input.GetImage(k, 0);
    for x = 1:ox:94    
        for y = 1:oy:94
            fprintf('x = %d, y = %d, k = %d\n', x, y, k);
            ForwardPass(X(:, :, x:(x + 130), y:(y + 130)), plan.layer{2});
            ix = (x - 1) / ox + 1;
            iy = (y - 1) / oy + 1;
            out(:, ix, iy, k) = plan.layer{end}.params.X;
%             t(1, ix, iy, k) = x;
%             t(2, ix, iy, k) = y;
        end
    end
end

out_train = out(:, :, :, 1:4);
t_train = out_train(:, 1:9, :, :) - out_train(:, 2:10, :, :);
out_train = out_train(:, 1:9, :, :);

out_test = out(:, :, :, 5);
t_test = out_test(:, 1:9, :, :) - out_test(:, 2:10, :, :);
out_test = out_test(:, 1:9, :, :);


out_train = out_train(:, :)';
t_train = t_train(:, :)';

out_test = out_test(:, :)';
t_test = t_test(:, :)';

theta_vec = pinv(out_train' * out_train) * out_train' * t_train;

q = out_test * theta_vec - t_test;
norm(q(:)) / norm(t_test(:))


% plot(svd(out(:, :)), '-r')
% hold on
% 
% outT = zeros(256, size(out, 2) - 1, size(out, 3));
% outT = out(:, 1:(end - 1), :) - out(:, 2:end, :);
% plot(svd(outT(:, :)), '-g')
% 
% 
% outE = zeros(256, size(out, 2) - 2, size(out, 3));
% outE = (out(:, 1:(end - 2), :) + out(:, 3:end, :)) / 2 - out(:, 2:(end - 1), :);
% plot(svd(outE(:, :)), '-b')
% legend('svd of f', 'svd of f(x_t+1) - f(x_t)', 'svd of llt');