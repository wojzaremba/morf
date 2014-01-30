% XXX : Execute on a single thread.
% clear all
addpath(genpath('../'));
global plan
json = ParseJSON('plans/imagenet.txt');
json{1}.batch_size = 128;
change = {};
for i = 1:length(json)
    if (strcmp(json{i}.type, 'Conv') == 1) 
        json{i}.type = 'ConvFFT';
        change{end + 1} = i;
    end
end

folds = 5;
times_fft = zeros(16, folds); 

Plan(json, 'trained/imagenet');
plan.training = 0;
for q = 1:length(change)
    pdims = plan.layer{change{q}}.prev_dim;
    W = plan.layer{change{q}}.params.W;
    W_ = zeros(pdims(1) + size(W, 2) - 1, pdims(2) + size(W, 3) - 1, size(W, 4), size(W, 1));
    for i = 1:size(W, 1)
        for j = 1:size(W, 4)
            W_(:, :, j, i) = fft2(squeeze(W(i, end:-1:1, end:-1:1, j)), size(W_, 1), size(W_, 2));
        end
    end
    plan.layer{change{q}}.params.W_ = W_;
end

for f = 0:folds
    fprintf('f = %d\n', f);
    plan.input.step = 1;
    plan.input.GetImage(0);
    ForwardPass(plan.input);
    if (f > 0)
        times_fft(:, f) = plan.time / plan.input.batch_size;
        fprintf('t = %f\n', sum(times_fft(:, f)));        
    end
    
correct = 0;    
for i = 1:json{1}.batch_size
    [~, idx] = sort(plan.classifier.params.out(i, :), 'descend');    
    for j = 1:5
        Y = plan.input.params.Y(i, :);
        if (idx(j) == find(Y(:)))
            correct = correct + 1;
        end
    end
end
fprintf('%d / %d\n', correct, json{1}.batch_size);    
    
end

times_fft128 = times_fft;
save('/Users/wojto/Dropbox/2013/compress/mat_files/eval_per_batch_128_batch_fft', 'times_fft128');
