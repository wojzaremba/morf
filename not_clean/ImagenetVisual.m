global plan
input = 'imagenet';
type = 'translation';
Init(['plans/', input, '.txt']);
load(['trained/', input], 'plan');
input_layer = plan.layer(1).v;
[oX, oY, plan.last_step] = input_layer.GetImage(9, 0);

plan.lr = 0;

layer = plan.layer(2).v;
plan.layer(end) = [];
lr = 0.001;
X = oX(1, :, :, :);
colormap('gray');
fprintf('Starting %s %s\n',  input, type);
translated = 0;
for k = 1:1000000 
    out = FP(X);    
    input = out;
    input(1, :) = input(1, :) - input(2, :);
    intut(2, :) = 0;    
    dparams = BP(input);
    ratio = norm(out(1, :) - out(2, :), 1) / norm(out(2, :), 1);
    fprintf('k = %d, ratio = %f, alpha = %f\n', k, ratio, translated);
        
    if (ratio > 0.01)
        layer.params.D = layer.params.D - lr * dparams.dD;
    else
        if (strcmp(type, 'translation') == 1)
            alpha = 12;
            translated = translated + alpha;
            layer.params.D(1, :) = layer.params.D(1, :) + translated;
        else
            assert(0);
        end        
    end        
    
    VisVecField(input, type, layer.params.D);
    img = permute([squeeze(oX(1, :, :, :)), squeeze(plan.layer(3).v.params.X(1, :, :, :)), squeeze(X(2, :, :, :))], [2, 3, 1]);
    img = min(max((img + 250) / 500, 0), 1);
    f = figure(1);
    set(f, 'Position', [300, 500, size(layer.params.D, 3), size(layer.params.D, 3) * 3]);
    imagesc(img);    
    drawnow;
end
