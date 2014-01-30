% clear all;
% global plan
% addpath(genpath('../'));
% Plan('plans/imagenet.txt', 'trained/imagenet');
% plan.training = 0;
% plan.input.ReloadData(1);

% for i = 1:length(plan.layer)
%     if (strcmp(plan.layer{i}.type, 'Conv'))
%         W = squeeze(plan.layer{i}.params.W(:, :));
%         [U, S, V] = svd(W);
%         D = diag(S);
%         approx = find(cumsum(D(end:-1:1)) / sum(D) < 0.01);
%         if (length(approx) == 0)
%             approx = size(W, 1);
%         else
%             approx = size(W, 1) - approx(end);
%         end
%         fprintf('Compressing %s layer by %f\n', plan.layer{i}.type, approx / size(W, 1));
% %         approx = floor(0.90 * size(W, 1));
% %         norm(U(:, 1:approx) * S(1:approx, :) * V' - W) / norm(W)
%         plan.layer{i}.params.newW = S(1:approx, :) * V';
%         plan.layer{i}.params.U = U(:, 1:approx);
%     end
% end


% correct = 0;
% totaltime = zeros(128, 1);
% for i = 1:128
%     [X, Y] = plan.input.GetImage(i, 0);
%     total = tic;
%     ForwardPass(X, plan.layer{2});
%     totaltime(i) = toc(total);
%     fprintf('Total time = %f\n', totaltime(i));
%     [~, idx] = sort(plan.layer{end}.out(1, :), 'descend');    
%     for j = 1:5
%         if (idx(j) == find(Y(1, :)))
%             correct = correct + 1;
%         end
%     end
%     fprintf('%d / %d\n', correct, i);
% end



% Common case
% layer = Conv, 0.108133
% layer = MaxPooling, 0.030838
% layer = LRNormal, 0.021115
% layer = Conv, 0.066620
% layer = MaxPooling, 0.007388
% layer = LRNormal, 0.025982
% layer = Conv, 0.022218
% layer = Conv, 0.029820
% layer = Conv, 0.024599
% layer = MaxPooling, 0.001024
% layer = FC, 0.233513
% layer = FC, 0.109228
% layer = FC, 0.017042
% layer = Softmax, 0.011101
% Total time = 0.389 +- 0.007
% correct = 110 / 128
