% compiles cuda code for monochromatic filter aproximation and tests for
% correctness.
clc;
clear all;
global root_path
init();

%flags
do_compile = 1;

gids.X = 1;
gids.X_mono = 2;
gids.W = 3;
gids.W_mono = 4;
gids.out = 5;
gids.out_mono = 6;
gids.perm = 7;
gids.colors = 8;

numImages = 128;
stride = 4;
padding = 0;
K = 11;
numFilters = 96;
numImgColors = 16;

if do_compile
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
    fid_read = fopen(strcat(root_path, 'expr_compress/cuda/src/monochromatic_input_template.cuh'), 'r');
    fid_write = fopen(strcat(root_path, 'expr_compress/cuda/src/monochromatic_input_gen.cuh'), 'wt');
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

    cd(strcat(root_path, 'expr_compress/cuda/'));
    status = system('make mexapprox');
    cd(root_path);

end
    
% Check correctness
X = randn(numImages, 224, 224, 3);
Wtmp = randn(numFilters, 3, K, K);
[W_approx, W_mono, colors, perm]  = monochromatic_approx(Wtmp, numImgColors);
out_ = single(zeros(numImages, 55, 55, numFilters));
out_mono_ = single(zeros(numImages, 55, 55, numFilters));

% copy to GPU for regular conv
C_(CopyToGPU, gids.W,  single(W_approx));
C_(CopyToGPU, gids.X,  single(X));
C_(CopyToGPU, gids.out,  out_);

C_(ConvAct, gids.X, gids.W, gids.out, size(X, 2), size(X, 4), size(W_approx, 2), stride, padding);
out = reshape(C_(CopyFromGPU, gids.out), size(out_));
C_(CleanGPU);


%copy to GPU for mono conv
Capprox_gen(CopyToGPU, gids.X,  single(X));
Capprox_gen(CopyToGPU, gids.W_mono,  single(W_mono));
Capprox_gen(CopyToGPU, gids.out_mono,  out_mono_);
Capprox_gen(CopyToGPU, gids.perm,  single(perm - 1));
Capprox_gen(CopyToGPU, gids.colors,  single(colors));
Capprox_gen(CopyToGPU, gids.X_mono,  single(zeros(numImages, 224, 224, numImgColors)));

Capprox_gen(Reshape, gids.X, 3, numImages * 224 * 224);
Capprox_gen(Mult, gids.X, gids.colors, gids.X_mono);
Capprox_gen(Reshape, gids.X_mono, 224*224*numImgColors, numImages);
Capprox_gen(approx_pointer, gids.X_mono, gids.W_mono, gids.out_mono, 224, numImgColors, K, stride, padding, gids.perm);

% Capprox_gen(approx_pointer, gids.X_mono, gids.W_mono, gids.out_mono, size(Xmono, 2), size(Xmono, 4), size(W_mono, 2), stride, padding, gids.perm);
out_mono = reshape(Capprox_gen(CopyFromGPU, gids.out_mono), size(out_mono_));
Capprox_gen(CleanGPU);

% are results equal?
assert(norm(out_mono(:) - out(:)) / norm(out(:)) < 1e-5);
