clear all;

W = randn(96, 3, 11, 11);
num_colors = 16;

% Approximate W
[Wapprox1, ~, ~, ~]  = monochromatic_approx(W, num_colors);
fprintf('||W - Wapprox1|| / ||W|| = %f\n', norm(W(:) - Wapprox1(:)) / norm(W(:)));

% approximate Wapprox1
[Wapprox2, Wm~ono, ~, ~]  = monochromatic_approx(Wapprox1, num_colors);
fprintf('||Wapprox1 - Wapprox2|| / ||Wapprox1|| = %f\n', norm(Wapprox1(:) - Wapprox2(:)) / norm(Wapprox1(:)));