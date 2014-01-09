A = single(randn(3, 5));
B_ = single(randn(31, 1));
C_(CopyToGPU, 1, A);
C_(CopyToGPU, 2, B_);
B = single(randn(3, 1));
C_(CopyToGPU, 4, B);
C_(CopyToGPU, 3, single(zeros(size(A))));
try
    C_(AddVector, 1, 2, 3);
    assert(false);
catch
    fprintf('Expected exception\n');
    assert(true);
end
C_(AddVector, 1, 4, 3);
X = C_(CopyFromGPU, 3);

assert(norm(A + repmat(B, [1, size(A, 2)]) - X) < 1e-4);

B = single(randn(1, 5));
C_(CopyToGPU, 2, B);
C_(AddVector, 1, 2, 3);
X = C_(CopyFromGPU, 3);

assert(norm(A + repmat(B, [size(A, 1), 1]) - X) < 1e-4);
