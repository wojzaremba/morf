global plan;
addpath(genpath('.'));
json = ParseJSON('plans/mnist_conv.txt');
json{1} = struct('type', 'VariableSizeInput', ...
                 'batch_size', 1000, ...
                 'rows', 28, ...
                 'cols', 28, ...
                 'file_pattern', 'data/mnist/', ...
                 'lambda', 3);
json{2}.bp_dx_on_first = 1;
Plan(json, 'trained/mnist_conv.mat');
plan.maxIter = 500;
plan.verbose = 10;
plan.training = 0;
plan.lr = 0;

while (plan.input.SetNewData())        
    minConf(@(opt_instances) distortfun(opt_instances), 0, 1);
    plan.input.cpu.vars.out = plan.input.badX;
    plan.input.cpu.vars.Y = plan.input.badY;
    plan.input.batch_size = size(plan.input.badX, 1);
    ForwardPass(plan.input);
    fprintf(1, 'Bad Label Accuracy: %g%%\n', plan.classifier.GetAcc());
    plan.input.active_indices = plan.classifier.CorrectIndices();
    plan.input.lambda = plan.input.lambda * 0.5;
    fprintf(1, 'Lambda: %g\n', plan.input.lambda);
end
