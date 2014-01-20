global root_path;
global plan;
clearvars -except root_path plan;
init();
load_imagenet_model();


% Replace first convolutional layer weights with approximated weights
args.num_colors = 8;
args.terms_per_element = 3; 
W = plan.layer{2}.cpu.vars.W;
[Wapprox, F, C, X, Y] = subspace_lowrank_approx(double(W), args);
plan.layer{2}.cpu.vars.W = Wapprox;


% Forward prop
plan.training = 0;
plan.input.step = 1;
plan.input.GetImage(0);
ForwardPass(plan.input);
plan.layer{2}.cpu.vars.W = W;

% Get error
test_error = plan.classifier.GetScore(5);
fprintf('errors = %d / %d\n', test_error, plan.input.batch_size);