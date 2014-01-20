%XXX : Why does uneven version fail?

clear all;

num_colors = 24;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. Test even version
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
W = randn(96, 11, 11, 3);

% Approximate W
[Wapprox1, ~, ~, ~]  = monochromatic_approx(W, struct('num_colors', num_colors, 'even', 1));
fprintf('||W - Wapprox1|| / ||W|| = %f\n', norm(W(:) - Wapprox1(:)) / norm(W(:)));

% approximate Wapprox1
[Wapprox2, ~, ~, ~]  = monochromatic_approx(Wapprox1, struct('num_colors', num_colors, 'even', 1));
fprintf('||Wapprox1 - Wapprox2|| / ||Wapprox1|| = %f\n', norm(Wapprox1(:) - Wapprox2(:)) / norm(Wapprox1(:)));
assert(norm(Wapprox1(:) - Wapprox2(:)) / norm(Wapprox1(:)) < 1e-10);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2. Test uneven version
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
W = randn(96, 11, 11, 3);

% Approximate W
[Wapprox1, ~, colors, ~]  = monochromatic_approx(W, struct('num_colors', num_colors, 'even', 0));
fprintf('||W - Wapprox1|| / ||W|| = %f\n', norm(W(:) - Wapprox1(:)) / norm(W(:)));

% approximate Wapprox1
[Wapprox2, ~, ~, ~]  = monochromatic_approx(Wapprox1, struct('num_colors', num_colors, 'even', 0));
fprintf('||Wapprox1 - Wapprox2|| / ||Wapprox1|| = %f\n', norm(Wapprox1(:) - Wapprox2(:)) / norm(Wapprox1(:)));
assert(norm(Wapprox1(:) - Wapprox2(:)) / norm(Wapprox1(:)) < 1e-10);




