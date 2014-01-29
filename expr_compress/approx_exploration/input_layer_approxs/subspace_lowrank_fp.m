global root_path;
global plan;
clearvars -except root_path plan;
init();
load_imagenet_model();

% Replace first convolutional layer weights with approximated weights
W = plan.layer{2}.cpu.vars.W;
args.num_clusters = 16;
args.terms_per_element = 3;
[Wapprox, ~, ~, ~, ~] = subspace_lowrank_approx(double(W), args);
plan.layer{2}.cpu.vars.W = Wapprox;


% Forward prop
error = 0;
plan.input.step = 1;
for i = 1:8
    plan.input.GetImage(0);
    ForwardPass(plan.input); 
    error = error + plan.classifier.GetScore(5);
    fprintf('%d / %d\n', error, i * plan.input.batch_size);
end

% Get error
test_error = plan.classifier.GetScore(5);
fprintf('errors = %d / %d\n', test_error, plan.input.batch_size);
plan.layer{2}.cpu.vars.W = W;