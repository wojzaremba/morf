% XXX :  Regress on two first layers.



global plan
addpath(genpath('../'));
SmallImagenetModel();

Y = zeros(128, 256);

% for k = 1:size(Y, 1)
k = 1
    fprintf('k = %d\n', k);
    [X, ~] = plan.input.GetImage(k, 0);
    ForwardPass(X(:, :, 1:131, 1:131), plan.layer{2});
    Y(k, :) = plan.layer{end}.params.X;
% end

% 
% allimg = zeros(6 * 131, 5 * 131, 3);
% 
% for i = 1:6
%     fprintf('i = %d\n', i);
%     [~, idx] = sort(Y(:, i), 'descend');
%     for j = 1:5         
%         fprintf('j = %d\n', j);
%         X = plan.input.GetImage(idx(j), 0);
%         X = squeeze(permute(X(1, :, 1:131, 1:131), [1, 3, 4, 2]) + 200) / 400;
%         allimg(((i - 1) * 131 + 1):(i * 131), ((j - 1) * 131 + 1):(j * 131), :) = X;
%     end
% end
% imagesc(allimg);
% 
% % ydata = tsne(Y');
% 
% for i=1:256
% hist(Y(:,i),64)
% pause
% end