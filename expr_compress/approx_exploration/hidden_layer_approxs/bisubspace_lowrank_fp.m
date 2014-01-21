global root_path;
global plan;
clearvars -except root_path plan;
init();
load_imagenet_model();


% Replace first convolutional layer weights with approximated weights
args.iclust = 12;
args.oclust = 32;
args.k = 12;
W = plan.layer{5}.cpu.vars.W;
[Wapprox, F, C, X, Y, assignment] = bisubspace_lowrank_approx(double(W), args);
plan.layer{5}.cpu.vars.W = Wapprox;


% Forward prop
plan.training = 0;
plan.input.step = 1;
plan.input.GetImage(0);
ForwardPass(plan.input);
plan.layer{5}.cpu.vars.W = W;

% Get error
error = 0;
plan.input.step = 1;
for i = 1:8
    plan.input.GetImage(0);
    ForwardPass(plan.input); 
    error = error + plan.classifier.GetScore(5);
    fprintf('%d / %d\n', error, i * plan.input.batch_size);
end