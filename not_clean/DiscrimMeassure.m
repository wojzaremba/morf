% clear all;
global plan
addpath(genpath('../'));
name = 'mnist_conv';
Plan(['plans/', name '.txt'], ['trained/', name]);
input = plan.input;
plan.training = 0;
input.ReloadData(10000);
[oXtest, oYtest, ~] = input.GetImage(1, 0);
fprintf('FP test\n');
ForwardPass(oXtest, plan.layer{2});
Xtest = {};
Xtrain = {};
types = {};
for i = 2:length(plan.layer)
    Xtest{end + 1} = sparse(plan.layer{i}.params.X(:, :));
    types{end + 1} = ['input_', plan.layer{i}.type];
    if (strcmp(plan.layer{i}.type, 'NNShared'))
        Xtest{end + 1} = sparse(plan.layer{i}.scratch.forward_act(:, :));
        types{end + 1} = ['act_', plan.layer{i}.type];
    end
end
Xtest{end + 1} = sparse(plan.layer{end}.out);
input.ReloadData(60000);
[oX, oY, ~] = input.GetImage(1, 1);
fprintf('FP train\n');
ForwardPass(oX, plan.layer{2});
for i = 2:length(plan.layer)
    Xtrain{end + 1} = sparse(plan.layer{i}.params.X(:, :));
    if (strcmp(plan.layer{i}.type, 'NNShared'))
        Xtrain{end + 1} = sparse(plan.layer{i}.scratch.forward_act(:, :));
    end
end
Xtrain{end + 1} = sparse(plan.layer{end}.out);

[~, Y] = max(oY, [], 2);
[~, Ytest] = max(oYtest, [], 2);
for i = 1:length(Xtrain)
    out5test = zeros(5, 1);
    out5 = zeros(5, 1);    
    Xcurr = Xtrain{i};
    Xtrain{i} = [];
    Xcurr_test = Xtest{i};
    Xtest{i} = [];
    model = train(Y, Xcurr, '-q -c 1');

    [out, ~, ~] = predict(ones(size(Y)), Xcurr, model, '-q');
    out_train = 1 - mean(out == Y);
    [out, ~, ~] = predict(ones(size(Ytest)), Xcurr_test, model, '-q');
    out_test = 1 - mean(out == Ytest);   
    fprintf('strip = %d/%d, output from layer name = %s, mean = %f, meantest = %f\n', i, length(Xtrain), types{i}, out_train, out_test);
end

