clear all;
global plan
Plan('plans/imagenet_matthew.txt', 'trained/imagenet_matthew');
plan.training = 0;

error = 0;
plan.input.step = 1;
plan.input.GetImage(2);

ForwardPass(plan.input); 
error = error + plan.classifier.GetScore(5);
fprintf('%d / %d\n', error, plan.input.batch_size);


% his : 55, 25, 12, 12, 6

% mine : 55, 25, 10, 9, 3
% mine_old : 55, 27, 13, 13, 6