% compiles cuda code for bisubspace clustering followed by low rank
% approximations of cluster tensors and tests for correctness and speedup. 
clear all;
global root_path
init();

if 1 % Just do this once.
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
gids.W = 2;
gids.out = 3;
gids.F = 4;
gids.C = 5;
gids.XY = 6;
gids.inPerm = 7;
gids.outPerm = 8;
gids.out_partial = 9;


N = 55;
M = 51;
numImages = 128;
stride = 1;
padding = 0;
K = 5;
Nout = 256;
Nin = 96;
numClustIn = 32;
numClustOut = 4;
rank = 8;
sizeClustIn = Nin / numClustIn;
sizeClustOut = Nout / numClustOut;

if do_compile
    % set cuda kernel vars
	cuda_vars.rank = rank;
	cuda_vars.numClustIn = numClustIn;
	cuda_vars.numClustOut = numClustOut;
	cuda_vars.sizeClustIn = sizeClustIn;
	cuda_vars.sizeClustOut = sizeClustOut;
	cuda_vars.clustersPerBlock = 1;
    cuda_vars.colorCache = 3;
    cuda_vars.B_Y = rank; 
    cuda_vars.B_X = 32;
    cuda_vars.imgsPerThread = 4; %numImages % 128 == 0 ? 4 : numImages % 6/4 == 0 ? 2 : 1;
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
F = randn(sizeClustOut, rank, numClustIn, numClustOut);
C = randn(sizeClustIn, rank, numClustIn, numClustOut);
XY = randn(K*K, rank, numClustIn, numClustOut);
out_partial = single(zeros(numImages, M, M, rank * numClustIn * numClustOut));
out_ = single(zeros(numImages, M, M, Nout));
inPerm = 0:(Nin - 1);
outPerm = 0:(Nout - 1);
num_runs = 10;

% copy to GPU for regular conv
C_(CopyToGPU, gids.W,  single(W));
C_(CopyToGPU, gids.X,  single(X));
C_(CopyToGPU, gids.out,  out_);


lapse1 = [];
for t=1:num_runs
	C_(StartTimer);
	C_(ConvAct, gids.X, gids.W, gids.out, size(X, 2), size(X, 4), size(W, 2), stride, padding);
	lapse = C_(StopTimer); 
	lapse1 = [lapse1, lapse];
end
out = reshape(C_(CopyFromGPU, gids.out), size(out_));
C_(CleanGPU);


% copy to GPU for approx conv
Capprox_gen(CopyToGPU, gids.F,  single(F));
Capprox_gen(CopyToGPU, gids.C,  single(C));
Capprox_gen(CopyToGPU, gids.XY,  single(XY));
Capprox_gen(CopyToGPU, gids.inPerm,  single(inPerm));
Capprox_gen(CopyToGPU, gids.outPerm,  single(outPerm));
Capprox_gen(CopyToGPU, gids.X,  single(X));
Capprox_gen(CopyToGPU, gids.out,  out_);
%Capprox_gen(CopyToGPU, gids.out_partial,  out_partial);

lapse2 = [];
for t=1:num_runs
	Capprox_gen(StartTimer);
	Capprox_gen(approx_pointer, gids.X, gids.F, gids.C, gids.XY, gids.out, gids.inPerm, gids.outPerm, N, Nin, Nout, K, stride, padding);
	lapse = Capprox_gen(StopTimer); 
	lapse2 = [lapse2, lapse];
end
out_approx = reshape(Capprox_gen(CopyFromGPU, gids.out), size(out_));
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
