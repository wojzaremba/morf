% clear all;
global plan
addpath(genpath('../'));
Plan('plans/imagenet.txt', 'trained/imagenet');
plan.training = 0;
plan.input.ReloadData(1);

[X, Y] = plan.input.GetImage(10, 0);
ForwardPass(X, plan.layer{2});
[~, idx] = sort(plan.layer{end}.out(1, :));
fprintf('\n\nTop 5 : ');
for i = 1:5
    fprintf('%d, ', idx(end - i));
end
fprintf('\n Correct answer : %d\n', find(Y(1, :)));

out = {};
for i = 2:12
    out{i} = plan.layer{i}.params.X;
end
% 
% SmallImagenetModel(12);
% 
% %%%%%%%%% Find all other 1
% %%%%%%%%%%%%%XXXXXXXXXXXX 
% 
% for of = 0:30
%     ForwardPass(X(:, :, (1 + of):(plan.layer{1}.dims(2) + of), (1 + of):(plan.layer{1}.dims(3) + of)), plan.layer{2});
%     out_new = plan.layer{end}.params.X;
% 
%     out_ = out{end};
%     out_ = out_(:, :, 2, 2);
%     fprintf('error = %f\n', norm(out_new(:) - out_(:)));
% end
