global plan
addpath(genpath('../../'));
load('../../trained/mnist_simple.mat');

[incr_test, err] = Test(0);
fprintf('incr_test = %d, err = %f\n', incr_test, err);

W = plan.layer{2}.cpu.vars.W;

[U, S, V] = svd(W);
s = 50;
W = U(:, 1:s) * S(1:s, 1:s) * V(:, 1:s)';
plan.layer{2}.cpu.vars.W = W;

[incr_test, err] = Test(0);
fprintf('approx incr_test = %d, err = %f\n', incr_test, err);