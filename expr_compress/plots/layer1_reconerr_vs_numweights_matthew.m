global root_path;
global plan;
clearvars -except root_path plan;
init();
load_imagenet_model();

W = plan.layer{2}.cpu.vars.W;

args.num_colors = 8;
args.even = 1;
args.start = 'sample';
num_colors = [1, 2, 3, 4, 6, 8, 12, 24, 48, 69]; 

num_weights = [];
recon_error = [];
test_error = [];
for i = 1:length(num_colors);
    args.num_colors = num_colors(i);
    [Wapprox, Wmono, colors, perm, nw] = monochromatic_approx(double(W), args);
    recon_error(i) = norm(W(:) - Wapprox(:));
    num_weights(i) = nw;
    plan.layer{2}.cpu.vars.W = Wapprox;
       
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
    plan.layer{2}.cpu.vars.W = W;
end


orig_num_weights = 96 * 256 * 5 * 5;