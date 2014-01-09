A = single(randn(10, 20, 30));
C_(CopyToGPU, 1, A);
C_(Scale, 1, single(4.), 1);
Q = C_(CopyFromGPU, 1);
assert(norm(Q(:) - 4 * A(:)) < 1e-4);
