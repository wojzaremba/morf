global root_path;
global plan;
clearvars -except root_path plan;
init();
load_imagenet_model();

W = plan.layer{5}.cpu.vars.W;

args.iclust = 32;
args.oclust = 8;
ranks = [4, 8, 12, 16, 20, 24, 28, 32, 64]; %[1, 2, 3, 4, 5, 6, 7, 8];

num_weights = [];
recon_error = [];
test_error = [];
for i = 1:length(ranks);
    args.k = ranks(i);
    [Wapprox, F, C, X, Y, assignment, nw] = bisubspace_lowrank_approx(double(W), args);
    recon_error(i) = norm(W(:) - Wapprox(:));
    num_weights(i) = nw;
    plan.layer{5}.cpu.vars.W = Wapprox;
       
    % Get error
    error = 0;
    plan.training = 0;
    plan.input.step = 1;
    for b = 1:8
        plan.input.GetImage(0);
        ForwardPass(plan.input); 
        error = error + plan.classifier.GetScore(5);
        fprintf('%d / %d\n', error, b * plan.input.batch_size);
    end
    test_error(i) = error;
    plan.layer{5}.cpu.vars.W = W;
end


orig_num_weights = 96 * 256 * 5 * 5;