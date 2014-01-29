% clear all;
global plan
addpath(genpath('.'));
name = 'mnist_conv';
Plan(['plans/', name '.txt'], ['trained/', name]);
plan.training = 0;

input = plan.input;
input.ReloadData(10000);
[oX, oY, last_step] = input.GetImage(1, 0);
fedge = input.dims(2);
edge = size(oX, 3);
oXT = zeros(size(oX, 1), size(oX, 2), fedge, fedge);
oXTR = zeros(size(oX, 1), size(oX, 2), fedge, fedge);
for i = 1:input.batch_size
    ox = (randi(3) - 2) * 3;
    oy = (randi(3) - 2) * 3;
    oXT(i, :, max(1 + ox, 1):min(edge + ox, edge), max(1 + oy, 1):min(edge + oy, edge)) = oX(i, :, max(1 - ox, 1):min(edge - ox, edge), max(1 - oy, 1):min(edge - oy, edge));
    ox = -ox;
    oy = -oy;
    oXTR(i, :, max(1 + ox, 1):min(edge + ox, edge), max(1 + oy, 1):min(edge + oy, edge)) = oX(i, :, max(1 - ox, 1):min(edge - ox, edge), max(1 - oy, 1):min(edge - oy, edge));
end

A = {};
AT = {};
ATR = {};
fprintf('FP1\n');
ForwardPass(oX, plan.layer{2});
for k = 2:length(plan.layer)
    A{k} = plan.layer{k}.params.X(:, :);      
end

fprintf('FP2\n');
ForwardPass(oXT, plan.layer{2});
for k = 2:length(plan.layer)        
    AT{k} = plan.layer{k}.params.X(:, :);      
end

fprintf('FP3\n');
ForwardPass(oXTR, plan.layer{2});
for k = 2:length(plan.layer)        
    ATR{k} = plan.layer{k}.params.X(:, :);      
end        

% Invariance meassure

for k = 2:length(plan.layer)        
    res_inv = zeros(5, 1);
    res_eqv = zeros(5, 1);    
    for q = 1:5
        A0 = A{k};
        A1 = AT{k};
        A2 = ATR{k};
        A0 = A0(((q - 1) * 2000 + 1):(q * 2000), :);
        A1 = A1(((q - 1) * 2000 + 1):(q * 2000), :);
        A2 = A2(((q - 1) * 2000 + 1):(q * 2000), :);   
        Y = oY(((q - 1) * 2000 + 1):(q * 2000), :);   
        res_inv(q) = NormalDist(A1 - A0, A0);
        res_eqv(q) = NormalDist((A1 + A2) / 2 - A0, A0);
    end
    fprintf('type = %s, norm = %f / %f, eq norm = %f / %f\n', plan.layer{k}.type, mean(res_inv), std(res_inv), mean(res_eqv), std(res_eqv));
end


A = A{end - 2};
many = 10;
units = 8;
allX = ones(29 * units, many * 28);
tmp = A(:, :) * randn(100, 100);
for u = 1:units
    [~, idx] = sort(tmp(:, u));
    X = permute(squeeze(oX(idx((end - many + 1):end), 1, :, :)), [2, 3, 1]);
    allX(((u - 1) * 29 + 1):(u * 29 - 1) , :) = X(:, :);
end
figure
colormap(gray);
imagesc(allX);

imwrite(allX, '/Users/wojto/Dropbox/2013/invariance_visualization/img/feature_vis_random_basis.png');