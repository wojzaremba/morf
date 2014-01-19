clear all
A = single(randn(3, 5));
B = single(zeros(3, 5));
C_(CopyToGPU, 1, A);
C_(CopyToGPU, 2, B);
C_(ActRELU, 1, 2);
B = C_(CopyFromGPU, 2);
assert(norm(max(A, 0) - B) < 1e-4);

C_(ActRELU, 1, 1);
A = C_(CopyFromGPU, 1);
assert(norm(double(A < 0)) == 0);