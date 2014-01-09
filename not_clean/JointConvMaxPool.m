global plan
addpath(genpath('../'));
Plan('plans/imagenet.txt', 'trained/imagenet');
plan.training = 0;
plan.input.ReloadData(1);

% plan.layer{2}.Fun = @LINEAR;
% 
% for i = 1:length(plan.layer)
%     try
%         plan.layer{i}.Fun;    
%         plan.layer{i}.Fun = @DROP;
%     catch
%     end
% end

correct = 0;
for i = 1:128
    [X, Y] = plan.input.GetImage(i, 0);
    ForwardPass(X, plan.layer{2});
    [~, idx] = sort(plan.layer{end}.out(1, :), 'descend');    
    for j = 1:5
        if (idx(j) == find(Y(1, :)))
            correct = correct + 1;
        end
    end
    fprintf('%d / %d\n', correct, i);
end