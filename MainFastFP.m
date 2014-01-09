% XXX : Extend number of test images (use other impl. of input layer).
% XXX : Functions to inspect content of this map of maps. 
% XXX : remove unnessesary funcs from Cmono.cu

% clear all
% global morf_path;
% morf_path = '/Users/Denton/Documents/morf-emily/';
% global plan
% addpath(genpath('.'));
% json = ParseJSON('/Users/Denton/Documents/morf-emily/plans/imagenet.txt');
% json{1}.batch_size = 128;
% Plan(json, '/Users/Denton/Documents/morf-emily/trained/imagenet');
% plan.training = 0;
% plan.input.step = 1;
% plan.input.GetImage(0);

S = Scheduler(struct('acceptance', 100, 'path', morf_path));
params = {};
numImgColors = [3, 4, 6];
for i = 1:3
    params{i}.numImgColors = numImgColors(i);
    params{i}.X =  plan.input.cpu.vars.out;
    params{i}.W = plan.layer{2}.cpu.vars.W;
end

cuda_params = {};
B_X = [32, 32, 32];
B_Y = [4, 4, 8];
imgsPerThread = [4, 4, 4];
filtersPerThread = [8, 6, 1]; % only relevant is numImgColors <= 4
colorsPerBlock = [1, 1, 1];
scale = [0, 0, 0];
checkImgBounds = [0, 0, 0]; % mod(numImages, B_X*imgsPerThread) ~= 0
for i = 1:3
    cuda_params{i}.B_X = B_X(i);
    cuda_params{i}.B_Y = B_Y(i);
    cuda_params{i}.imgsPerThread = imgsPerThread(i);
    cuda_params{i}.filtersPerThread = filtersPerThread(i);
    cuda_params{i}.scale = scale(i);
    cuda_params{i}.checkImgBounds = checkImgBounds(i);
    cuda_params{i}.colorsPerBlock = colorsPerBlock(i);
end

S.Add(@monochromatic_input, params, cuda_params);
S.Run();
display_scheduler_maps(S.funcs, S.params_map);