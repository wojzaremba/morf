% compiles cuda code for monochromatic filter aproximation and tests for
% correctness and speedup.

clear all;
morf_path = '/Volumes/denton/Documents/morf/';
addpath(genpath(morf_path));

%flags
compile = 1;
correctness = 1;
speedup = 1;


gids.X = 1;
gids.X_mono = 2;
gids.W = 3;
gids.W_mono = 4;
gids.out = 5;
gids.out_mono = 6;
gids.perm = 7;

numImages = 128;
stride = 4;
padding = 0;
K = 11;
numFilters = 96;
numImgColors = 4;
perm = [0, 1, 2];

if compile
% set cuda kernel vars
filtersPerColor = numFilters / numImgColors;
filtersPerThread = 4; % only relevant if numImgColors <= 4
B_Y = 6;
B_X = 32;
colorsPerBlock = 1;%filtersPerThread * B_Y / filtersPerColor;
imgsPerThread = 4; %numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
scale = 0;
checkImgBounds = mod(numImages, B_X*imgsPerThread) ~= 0;

% replace template variables, and compile
fid_read = fopen(strcat(morf_path, 'expr_compress/cuda/src/monochromatic_input_template.cuh'), 'r');
fid_write = fopen(strcat(morf_path, 'expr_compress/cuda/src/monochromatic_input_gen.cuh'), 'wt');
line = fgets(fid_read);
while ischar(line)
    %disp(rline);
    line = strrep(line, '#B_Y', num2str(B_Y));
    line = strrep(line, '#B_X', num2str(B_X));
    line = strrep(line, '#imgsPerThread', num2str(imgsPerThread));
    line = strrep(line, '#filtersPerThread', num2str(filtersPerThread));
    line = strrep(line, '#colorsPerBlock', num2str(colorsPerBlock));
    line = strrep(line, '#scale', num2str(scale));
    line = strrep(line, '#checkImgBounds', num2str(checkImgBounds));
    line = strrep(line, '\n', '\\n');
    line = strrep(line, '%', '%%');
    fprintf(fid_write, line);
    line = fgets(fid_read);
end

fclose(fid_read);
fclose(fid_write);

cd(strcat(morf_path, 'expr_compress/cuda/'));
status = system('make mexapprox');

end

if correctness
    
% first check correctness
X_mono = single(randn(numImages, 224, 224, numImgColors));
X = X_mono;
out_ = single(zeros(numImages, 55, 55, numFilters));
out_mono_ = single(zeros(numImages, 55, 55, numFilters));
W_mono = single(ones(numFilters, K, K));
W = single(zeros(numFilters, K, K, numImgColors));

bpt = 0:(numFilters/numImgColors):numFilters;
for i=1:numImgColors
   W_mono( bpt(i)+1:bpt(i+1), :, : ) = single(randn(bpt(i+1) - bpt(i), K, K));
   W( bpt(i)+1:bpt(i+1), :, :, i) = W_mono( bpt(i)+1:bpt(i+1), :, : );
end

perm = randperm(numFilters) - 1;

W_mono = W_mono(perm+1, :, :);

% copy to GPU for regular conv
C_(CopyToGPU, gids.W,  W);
C_(CopyToGPU, gids.X,  X);
C_(CopyToGPU, gids.out,  out_);

C_(ConvAct, gids.X, gids.W, gids.out, size(X, 2), size(X, 4), size(W, 2), stride, padding);
out = reshape(C_(CopyFromGPU, gids.out), size(out_));
C_(CleanGPU);


%copy to GPU for mono conv
Capprox_gen(CopyToGPU, gids.W_mono,  W_mono);
Capprox_gen(CopyToGPU, gids.X_mono,  X_mono);
Capprox_gen(CopyToGPU, gids.out_mono,  out_mono_);
Capprox_gen(CopyToGPU, gids.perm,  single(perm));

Capprox_gen(approx_pointer, gids.X_mono, gids.W_mono, gids.out_mono, size(X_mono, 2), size(X_mono, 4), size(W_mono, 2), stride, padding, gids.perm);
out_mono = reshape(Capprox_gen(CopyFromGPU, gids.out_mono), size(out_mono_));
Capprox_gen(CleanGPU);

% are results equal?
eq = sum(out_mono(:) ~= out(:));
if eq
    fprintf('Monochromatic conv output is incorrect\n');
end 

end

if speedup
    
% now check runtime
X_mono = single(randn(numImages, 224, 224, numImgColors));
X = single(randn(numImages, 224, 224, 3));
out_ = single(zeros(numImages, 55, 55, numFilters));
out_mono_ = single(zeros(numImages, 55, 55, numFilters));
W = single(randn(numFilters, K, K, 3));
W_mono = single(zeros(numFilters, K, K));

num_runs = 10;

% copy to GPU for regular conv
C_(CopyToGPU, gids.X,  X);
C_(CopyToGPU, gids.W,  W);
C_(CopyToGPU, gids.out,  out_);
lapse1 = [];
for t=1:num_runs
    C_(StartTimer);
    C_(ConvAct, gids.X, gids.W, gids.out, size(X, 2), size(X, 4), size(W, 2), stride, padding);
    lapse = C_(StopTimer); 
    out = reshape(C_(CopyFromGPU, gids.out), size(out_));
    lapse1 = [lapse1, lapse];
end
C_(CleanGPU);

% copy to GPU for mono conv
Capprox_gen(CopyToGPU, gids.X_mono,  X_mono);
Capprox_gen(CopyToGPU, gids.W_mono,  W_mono);
Capprox_gen(CopyToGPU, gids.out_mono,  out_mono_);
Capprox_gen(CopyToGPU, gids.perm,  perm);

lapse2 = [];
for t=1:num_runs
    Capprox_gen(StartTimer);
    Capprox_gen(approx_pointer, gids.X_mono, gids.W_mono, gids.out_mono, size(X_mono, 2), size(X_mono, 4), size(W_mono, 2), stride, padding, gids.perm);
    lapse = Capprox_gen(StopTimer); 
    out_mono = reshape(Capprox_gen(CopyFromGPU, gids.out_mono), size(out_));
    lapse2 = [lapse2, lapse];
end
Capprox_gen(CleanGPU);

speedup = lapse1 ./ lapse2;
fprintf('average speedup = %f\n', mean(speedup));
fprintf('std speedup = %f\n', std(speedup));

end