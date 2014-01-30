global plan debug
debug = 2;
W = plan.layer{5}.cpu.vars.W;

WW = plan.layer{5}.cpu.vars.W;
WW = permute(WW, [1, 4, 2, 3]);
WW = sum(WW(:, :, :).^2, 3);
imagesc(WW);

mask = WW > 0.004;

mask = repmat(reshape(mask, [size(mask, 1), 1, 1, size(mask, 2)]), [1, 5, 5, 1]);

plan.layer{5}.cpu.vars.W = plan.layer{5}.cpu.vars.W .* mask;
plan.layer{5}.cpu.vars.W = single(plan.layer{5}.cpu.vars.W);

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
% 
% W = double(W);
% 
inter = size(W, 1);
A = eye(size(W, 1), inter);
% 
% 
% 
% 
WS = sparse(W(:, :));
cvx_begin   
    variable WWW(inter, size(W, 2), size(W, 3), size(W, 4))
    minimize( norm( A * WWW(:, :) - WS, 2 ))
cvx_end
%  norm(WWW(:), 2) )
%  +
% 
% 
% cvx_begin   
%     variable A(size(W, 1), inter)
%     minimize( norm( A * WWW(:, :) - W(:, :), 2 ))
% cvx_end