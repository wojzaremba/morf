clear all;

args.iclust = 12;
args.oclust = 32;
args.k = 2;
W = randn(256, 5, 5, 96);

% Approximate W
[Wapprox1, ~, ~, ~, ~, ~]  = bisubspace_lowrank_approx(W, args);
fprintf('||W - Wapprox1|| / ||W|| = %f\n', norm(W(:) - Wapprox1(:)) / norm(W(:)));

% approximate Wapprox1
[Wapprox2, ~, ~, ~, ~, ~]  = bisubspace_lowrank_approx(Wapprox1, args);
fprintf('||Wapprox1 - Wapprox2|| / ||Wapprox1|| = %f\n', norm(Wapprox1(:) - Wapprox2(:)) / norm(Wapprox1(:)));
assert(norm(Wapprox1(:) - Wapprox2(:)) / norm(Wapprox1(:)) < 1e-10);
