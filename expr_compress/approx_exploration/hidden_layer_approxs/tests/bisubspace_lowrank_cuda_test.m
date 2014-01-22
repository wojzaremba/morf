% compiles cuda code for bisubspace clustering followed by low rank
% approximations of cluster tensors and tests for correctness and speedup. 
clear all;
global root_path
init();

if 0 % Just do this once.
    % replace #mock with, func_name 
    fid_read = fopen(strcat(root_path, '/expr_compress/cuda/src/Capprox_template.cu'), 'r');
    fid_write = fopen(strcat(root_path, '/expr_compress/cuda/src/Capprox_gen.cu'), 'wt');
    line = fgets(fid_read);
    while ischar(line)
        line = strrep(line, '#mock' , 'biclustered_hidden'); 
        line = strrep(line, '\n', '\\n');
        line = strrep(line, '%', '%%');
        line = strrep(line, '\', '\\');
        fprintf(fid_write, line);
        line = fgets(fid_read);
    end
    fclose(fid_read);
    fclose(fid_write);
end          
%flags
do_compile = 1;

gids.X = 1;

gids.W = 3;

gids.out = 5;



N = 55;
numImages = 128;
stride = 1;
padding = 0;
K = 5;
numOut = 256;
numIn = 96;

if do_compile
    % set cuda kernel vars
    cuda_vars.colorCache = 2;
    cuda_vars.filtersPerThread = 4; % only relevant if newNumColors <= 4
    cuda_vars.B_Y = 4;
    cuda_vars.B_X = 32;
    cuda_vars.imgsPerThread = 4; %numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
    cuda_vars.scale = 0;
    cuda_vars.checkImgBounds = mod(numImages, cuda_vars.B_X*cuda_vars.imgsPerThread) ~= 0;

    % replace template variables, and compile
    fid_read = fopen(strcat(root_path, 'expr_compress/cuda/src/biclustered_hidden_template.cuh'), 'r');
    fid_write = fopen(strcat(root_path, 'expr_compress/cuda/src/biclustered_hidden_gen.cuh'), 'wt');
    line = fgets(fid_read);
    while ischar(line)
        fields = fieldnames(cuda_vars);
        for f = 1:length(fields)
           line = strrep(line, strcat('#', fields{f}), num2str(getfield(cuda_vars, fields{f}))); 
        end
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
X = ones(numImages, N, N, Nin);
W = randn(Nout, K, K, Nin);
out_ = single(zeros(numImages, 51, 51, Nout));

num_runs = 0;


% copy to GPU for regular conv
Capprox_gen(CopyToGPU, gids.W,  single(W_approx));
Capprox_gen(CopyToGPU, gids.X,  single(X));
Capprox_gen(CopyToGPU, gids.out,  out_);
% 
% 
% lapse1 = [];
% for t=1:num_runs
%     C_(StartTimer);
     Capprox_gen(ConvAct, gids.X, gids.W, gids.out, size(X, 2), size(X, 4), size(W_approx, 2), stride, padding);
%     lapse = C_(StopTimer); 
%     lapse1 = [lapse1, lapse];
% end
% out = reshape(C_(CopyFromGPU, gids.out), size(out_));
 Capprox_gen(CleanGPU);
% 
% 
% 
% % are results equal?
% assert(norm(out_mono(:) - out(:)) / norm(out(:)) < 1e-5);
% 
% speedup = lapse1 ./ lapse2;
% fprintf('average speedup = %f\n', mean(speedup));
% fprintf('std speedup = %f\n', std(speedup));
