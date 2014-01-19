times = zeros(5, 2);
batch_size = 100;
input_size = 4000;
output_size = 1000;
eigen_vals = 20;
for i = 1:size(times, 1)
    fprintf('i = %d\n', i);
    X = randn(batch_size, input_size);
    W = randn(input_size, output_size);
   
    [U, S, V] = svd(W);
    S(eigen_vals:end, eigen_vals:end) = S(eigen_vals:end, eigen_vals:end) / 100000; 
    W = U * S * V';
    [U, S, V] = svd(W);
    U = U(:, 1:eigen_vals);
    S = S(1:eigen_vals, :);
    Wapprox = U * S * V';
    assert(norm(W(:) - Wapprox(:)) / norm(W(:)) < 1e-3);

    tic;
    X * W;
    times(i, 1) = toc;

    tic;
    ((X * U) * S) * V';
    times(i, 2) = toc;
end