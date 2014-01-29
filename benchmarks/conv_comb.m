clear all
global plan
addpath(genpath('..'));
json = ParseJSON('plans/imagenet.txt');
json{1}.batch_size = 128;
Plan(json, 'trained/imagenet');
plan.training = 0;
plan.input.step = 1;
plan.input.GetImage(0);

L = 24; %number of clusters
f = 2; %number of terms per element

lambda = 0.001;
W = double(plan.layer{2}.cpu.vars.W);
CMat = SparseCoefRecovery(W(:, :)', 0,'Lasso', lambda);
CKSym = BuildAdjacency(CMat, 0);
[Grps] = SpectralClustering(CKSym, L);
order= Grps(:,2);

[ordsort,popo]=sort(order);
%compress each cluster
for l = 1 : L
    I = find(order == l);
    if ~isempty(I)
    chunk = W(I, :, :, :);
        [C{l}, H{l}, V{l}, F{l}, cappr] = rankoneconv(chunk, f * length(I));
        layer_appr(I, :, :, :) = cappr;
    end
end

plan.layer{2}.cpu.vars.W = single(layer_appr);
ForwardPass(plan.input);
fprintf('errors = %d / %d\n', plan.classifier.GetScore(5), json{1}.batch_size);