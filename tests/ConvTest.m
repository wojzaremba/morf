images = single(randn([128, 8, 8, 32]));
filters = single(randn([64, 8, 8, 32]));
targets = single(zeros([128, 1, 1, 64]));
C_(CopyToGPU, 1, images(:, :));
C_(CopyToGPU, 2, filters(:, :));
C_(CopyToGPU, 3, targets(:, :));

stride = 1;
padding = 0;
C_(ConvAct, 1, 2, 3, size(images, 2), size(images, 4), size(filters, 2), stride, padding);

Q = C_(CopyFromGPU, 3);

assert(norm(sum(images(5, :) .* filters(7, :)) - Q(5, 7)) < 1e-4);