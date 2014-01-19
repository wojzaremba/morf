global root_path;
global plan;
clearvars -except root_path plan;
init();
load_imagenet_model();

% Replace first convolutional layer weights with approximated weights
num_colors = 6;
W = plan.layer{2}.cpu.vars.W;
[Wapprox, Wmono, colors, perm] = monochromatic_approx(double(W), struct('num_colors', num_colors, 'even', 0));
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