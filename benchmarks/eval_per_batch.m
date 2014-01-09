% Execute with option singleCompThread !!!

clear all;
addpath(genpath('../'));
global plan
json = ParseJSON('plans/imagenet.txt');

folds = 5;
timess = zeros(16, folds, 10); 
for i = 0:9
    json{1}.batch_size = 2 ^ i;
    Plan(json, 'trained/imagenet');
    plan.training = 0;    
    for f = 0:folds 
        fprintf('i = %d, f = %d\n', i, f);
        plan.input.step = 1;
        plan.input.GetImage(0);        
        ForwardPass(plan.input);
        if (f > 0)
            timess(:, f, i + 1) = plan.time / plan.input.batch_size;
            fprintf('t = %f\n', sum(timess(:, f, i + 1)));            
        end
    end
end
save('/Users/wojto/Dropbox/2013/compress/mat_files/eval_per_batch', 'timess');

