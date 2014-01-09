% addpath(genpath('../'));
images = single(randn([1, 8, 8, 32]));
targets = single(zeros([1, 8, 8, 32]));
C_(CopyToGPU, 1, images(:, :));
C_(CopyToGPU, 2, targets(:, :));

stride = 1;
poolsize = 2;
C_(MaxPool, 1, 2, size(images, 4), poolsize, stride, size(filters, 2));

Q = C_(CopyFromGPU, 2);

assert(sum(Q(:) == images(:)) > length(Q(:)) / 4);