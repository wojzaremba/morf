clc;
global debug plan
debug = 2;
init();
load_mock_model();


num_image_colors = 16;
dims = [96, 11, 11, 3];
colors = randn([num_image_colors, dims(4)]);
dec = randn(dims(1), dims(4));
Wmono = randn([dims(1), dims(2), dims(3)]);
S = randn(dims(1), dims(2)*dims(3));
assignment = reshape(repmat(1:num_image_colors', [1, dims(1) / num_image_colors]), dims(1), 1);
W = MonochromaticInput.ReconstructW(colors, dec, S, assignment, [dims(1), dims(4), dims(2), dims(3)]);
plan.layer{2}.cpu.vars.W = W;
approx_params = struct('num_image_colors', num_image_colors);
approx = MonochromaticInput('_test',  struct(), struct());
[Wapprox, ret] = approx.Approx(approx_params);

assert(norm(W(:) - Wapprox(:)) < 1e-4);

