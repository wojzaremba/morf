clear all;

args.num_colors = 48;
args.terms_per_element = 2;
W = randn(96, 11, 11, 3);

% Approximate W
[Wapprox1, ~, ~, ~, ~]  = subspace_lowrank_approx(W, args);
fprintf('||W - Wapprox1|| / ||W|| = %f\n', norm(W(:) - Wapprox1(:)) / norm(W(:)));

% approximate Wapprox1
[Wapprox2, ~, ~, ~, ~]  = subspace_lowrank_approx(Wapprox1, args);
fprintf('||Wapprox1 - Wapprox2|| / ||Wapprox1|| = %f\n', norm(Wapprox1(:) - Wapprox2(:)) / norm(Wapprox1(:)));
assert(norm(Wapprox1(:) - Wapprox2(:)) / norm(Wapprox1(:)) < 1e-10);
