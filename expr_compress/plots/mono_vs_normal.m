% 1. Rearrange memory.
% 2. Count number of memory accesses.
% 3. Make sure that I am accessing consequtive pieces of memory all the
% time.


global root_path plan;
json_old = ParseJSON('plans/imagenet_matthew.txt');
json = {};
json{1} = json_old{1};
json{2} = json_old{2};
        
json{1}.batch_size = 1;
% Plan(json, '~/imagenet_data/imagenet_matthew', 0, 'single');
Plan(json, [], 0, 'single');
plan.training = 0;
plan.input.step = 1;
plan.input.GetImage(0);

plan.layer{2}.cpu.vars.X = plan.layer{1}.cpu.vars.out;
tic;
plan.layer{2}.FPcpp();
fprintf('common conv takes = %f\n', toc);
outFPcommon = plan.layer{2}.cpu.vars.out;


json{2}.type = 'MonoConv';
json{2}.num_image_colors = 12;
Plan(json, [], 0, 'single');
plan.training = 0;
plan.input.step = 1;
plan.input.GetImage(0);

plan.layer{2}.cpu.vars.X = single(plan.layer{1}.cpu.vars.out);
plan.layer{2}.cpu.vars.Cmono = single(randn(size(plan.layer{2}.cpu.vars.Cmono)));
plan.layer{2}.cpu.vars.Wmono = single(randn(size(plan.layer{2}.cpu.vars.Wmono)));
plan.layer{2}.cpu.vars.perm = single(randperm(seed, json{2}.depth)' - 1);

tic;
plan.layer{2}.FPcpp();
fprintf('mono conv takes = %f\n', toc);
outFPmono = plan.layer{2}.cpu.vars.out;


% New plan.

% outFPcpp = plan.layer{2}.cpu.vars.out;



% [Wapprox, Wmono, colors, perm] = monochromatic_approx(double(W), struct('num_colors', num_colors, 'even', 1, 'start', 'sample'));


% % Replace first convolutional layer weights with approximated weights
% num_colors = 8;

% plan.layer{2}.cpu.vars.W = Wapprox;
% 
% 
% % Forward prop
% error = 0;
% plan.input.step = 1;
% for i = 1:8
%     plan.input.GetImage(0);
%     ForwardPass(plan.input); 
%     error = error + plan.classifier.GetScore(5);
%     fprintf('%d / %d\n', error, i * plan.input.batch_size);
% end
% 
% % Get error
% test_error = plan.classifier.GetScore(5);
% fprintf('errors = %d / %d\n', test_error, plan.input.batch_size);
% plan.layer{2}.cpu.vars.W = W;
