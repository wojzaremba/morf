global plan
addpath(genpath('../'));
layer_nr = 11;
SmallImagenetModel(layer_nr);

feature = 1;
dims = plan.layer{layer_nr}.dims;
maxval = zeros(dims(2), dims(3), 20);
fprintf('Deconvolution for %s\n', plan.layer{layer_nr}.name);
for k = 1:20
    fprintf('k = %d\n', k);
    [X, ~] = plan.input.GetImage(k, 0);
    ForwardPass(X(:, :, 1:131, 1:131), plan.layer{2});
    maxval(:, :, k) = squeeze(plan.layer{layer_nr + 1}.params.X(1, feature, :, :));
end
[val, idx] = max(maxval(:));
ox = mod(idx - 1, size(maxval, 1)) + 1;
oy = mod(floor((idx - 1) / size(maxval, 1)), size(maxval, 2)) + 1;
ok = floor((idx - 1) / (size(maxval, 1) * size(maxval, 2))) + 1;



act = zeros(size(plan.layer{layer_nr + 1}.params.X));
act(1, feature, ox, oy) = val;
[X, ~] = plan.input.GetImage(ok, 0);
ForwardPass(X, plan.layer{2});
for k = layer_nr:-1:2
    act = plan.layer{k}.Invert(act);
end
X = permute(squeeze(act), [2, 3, 1]);
X = X - min(X(:));
X = X / max(X(:));
imagesc(X);


