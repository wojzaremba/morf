% compiles cuda code for monochromatic filter approximation (with SVD) and tests for
% correctness.
global root_path plan
init;
load_imagenet_model;

name = 'monolowrank_input';
if 0 % Just do this once.
    % replace #mock with, func_name 
    fid_read = fopen(strcat(root_path, '/expr_compress/cuda/src/Capprox_template.cu'), 'r');
    fid_write = fopen(strcat(root_path, '/expr_compress/cuda/src/Capprox_gen.cu'), 'wt');
    line = fgets(fid_read);
    while ischar(line)
        line = strrep(line, '#mock' , name); 
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
gids.Wapprox = 2;
gids.out = 3;
gids.out_approx = 4;
gids.colors = 5;
gids.filters = 6;
gids.filter_coeff = 7;
gids.perm = 8;

numImages = 128;
stride = 2;
padding = 0;
K = 7;
numFilters = 96;

if do_compile
    % set cuda kernel vars
    cuda_vars.numImgColors = 3;
    cuda_vars.numClusters = 3;
    cuda_vars.clusterSize = numFilters / cuda_vars.numClusters;
    cuda_vars.rank = 6;
    cuda_vars.filtersPerThread = 8; 
    cuda_vars.B_Y = 4;
    cuda_vars.B_X = 32;
    cuda_vars.imgsPerThread = 4; %numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
    cuda_vars.scale = 0;
    cuda_vars.checkImgBounds = mod(numImages, cuda_vars.B_X*cuda_vars.imgsPerThread) ~= 0;

    % replace template variables, and compile
    fid_read = fopen(strcat(root_path, 'expr_compress/cuda/src/', name, '_template.cuh'), 'r');
    fid_write = fopen(strcat(root_path, 'expr_compress/cuda/src/', name, '_gen.cuh'), 'wt');
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
    

% Test correctness
% Check correctness
X = ones(numImages, 224, 224, 3);
Wtmp = randn(numFilters, K, K, 3);
args.rank = cuda_vars.rank;
args.num_clusters = cuda_vars.numClusters;
[Wapprox, colors, filters, filter_coeff, perm] = monochromatic_lowrank_approx(Wtmp, args);
out_ = single(zeros(numImages, 110, 110, numFilters));
out_approx_ = single(zeros(numImages, 110, 110, numFilters));
perm = 0:95;
num_runs = 1;

%copy to GPU for mono conv
Capprox_gen(CopyToGPU, gids.X,  single(X));
Capprox_gen(CopyToGPU, gids.Wapprox,  single(Wapprox));
Capprox_gen(CopyToGPU, gids.out_approx,  out_approx_);
Capprox_gen(CopyToGPU, gids.colors,  single(colors));
Capprox_gen(CopyToGPU, gids.filters,  single(filters));
Capprox_gen(CopyToGPU, gids.filter_coeff,  single(filter_coeff));
Capprox_gen(CopyToGPU, gids.perm,  single(perm));

lapse2 = [];
for t=1:num_runs
    Capprox_gen(StartTimer);
    Capprox_gen(approx_pointer, gids.X, gids.out_approx, gids.colors, gids.filters, gids.filter_coeff, gids.perm, size(X, 2), size(X, 4), K, stride, padding);
    lapse = Capprox_gen(StopTimer); 
    lapse2 = [lapse2, lapse];
end

out_approx = reshape(Capprox_gen(CopyFromGPU, gids.out_approx), size(out_approx_));
Capprox_gen(CleanGPU);

% copy to GPU for regular conv
C_(CopyToGPU, gids.Wapprox,  single(Wapprox));
C_(CopyToGPU, gids.X,  single(X));
C_(CopyToGPU, gids.out,  out_);


lapse1 = [];
for t=1:num_runs
    C_(StartTimer);
    C_(ConvAct, gids.X, gids.Wapprox, gids.out, size(X, 2), size(X, 4), size(Wapprox, 2), stride, padding);
    lapse = C_(StopTimer); 
    lapse1 = [lapse1, lapse];
end
out = reshape(C_(CopyFromGPU, gids.out), size(out_));
C_(CleanGPU);

fprintf('||out - out_approx|| = %f\n\n', norm(out(:) - out_approx(:)));

speedup = lapse1 ./ lapse2;
fprintf('average speedup = %f\n', mean(speedup));
fprintf('std speedup = %f\n', std(speedup));