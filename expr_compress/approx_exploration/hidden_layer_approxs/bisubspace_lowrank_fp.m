global root_path;
global plan;
clearvars -except root_path plan;
init();
load_imagenet_model(); % By default, loads Matt's model.


% Replace first convolutional layer weights with approximated weights
args.iclust = 32;
args.oclust = 16;
args.k = 8;
args.cluster_type = 'subspace';
W = plan.layer{5}.cpu.vars.W;
  
[Wapprox, ~, ~, ~, ~, ~, ~] = bisubspace_lowrank_approx(double(W), args);
plan.layer{5}.cpu.vars.W = Wapprox;


% Get error
error = 0;
plan.input.step = 1;
for i = 1:8
    plan.input.GetImage(0);
    ForwardPass(plan.input); 
    error = error + plan.classifier.GetScore(5);
    fprintf('%d / %d\n', error, i * plan.input.batch_size);
end

plan.layer{5}.cpu.vars.W = W;
