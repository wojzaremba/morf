% clear all
global cuda
cuda = zeros(1000, 1);
addpath(genpath('../'));
Q = randn(10, 20);
CopyToGPU_(1, 10, Q);

Q = Q + 10;
X = CopyFromGPU_(1, 10);

% global plan
% jsons = {};
% jsons{1} = struct('depth', 4, 'rows', 8, 'cols', 10, 'batch_size', 6, 'type', 'TestInput');
% jsons{2} = struct('type', 'NNShared', 'local_2d_patch', struct('patch_rows', 4, 'patch_cols', 4, 'stride_rows', 2, 'stride_cols', 2, 'padding_rows', 1, 'padding_cols', 1), 'depth', 5, 'function', 'RELU');
% Plan(jsons);
% 
% X = plan.input.GetImage(1, 1);
% CopyToGPU_(2, 0, X);
% plan.layer{2}.FP_cuda();