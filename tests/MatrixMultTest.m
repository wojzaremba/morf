clear all
A = single(randn(10, 20));
B = single(randn(20, 5));
C_(CopyToGPU, 1, A);
C_(CopyToGPU, 2, B);
C_(CopyToGPU, 3, single(zeros(size(A, 1), size(B, 2))));
C_(Mult, 1, 2, 3);
X = C_(CopyFromGPU, 3);
assert(norm(X - A * B) < 1e-4);
