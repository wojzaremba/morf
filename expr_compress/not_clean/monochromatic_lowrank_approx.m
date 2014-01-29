global plan root_path
init();
%type = 'alex';
%load_imagenet_model(type);
load_imagenet_model();
Worig = plan.layer{2}.cpu.vars.W;
W = plan.layer{2}.cpu.vars.W;
K = 7;

W = permute(W, [1, 4, 2, 3]);
approx0  =zeros(size(W));
for f = 1 : size(W,1)
    [u,s,v]=svd(squeeze(W(f,:,:)),0);
    C(f, :) = u(:, 1);
    S(f, :) = s(1, 1) * v(:, 1); %v(:,1);
    chunk = u(:, 1) * s(1, 1) * v(:, 1)';
    Wapprox0(f, :, :, :) = reshape(chunk, 1, size(W, 2), size(W, 3), size(W, 4));
end

 num_colors = 12;
[assignment,colors] = litekmeans(C', num_colors);
colors = colors';

Wapprox1 = zeros(size(W));
for f=1:size(W,1)
    chunk = (colors(assignment(f),:)') * (S(f,:));
    Wapprox1(f,:,:,:)=reshape(chunk,1,size(W,2),size(W,3),size(W,4));
end 

% Now approximate within each cluster
rank = 8;
Wmono = reshape(S,size(W,1),size(W,3),size(W,4));
Wmonoapprox = zeros(size(Wmono));
for l = 1 : num_colors
    I = find(assignment == l);
    if ~isempty(I)
        chunk=Wmono(I, :, :);
        [u, s, v] = svd(chunk(:, :));
        cappr = u(:, 1:rank) *s(1:rank, 1:rank) * v(:, 1:rank)';
        Wmonoapprox(I, :)=cappr;
    end
end

Wapprox2 = zeros(size(W));
for f=1:size(W,1)
    chunk = (C(f,:)') * (Wmonoapprox(f,:));
    Wapprox2(f,:,:,:)=reshape(chunk,1,size(W,2),size(W,3),size(W,4));
end 

collage_imnet(W);
collage_imnet(Wapprox2);

Wapprox = permute(Wapprox2, [1, 3, 4, 2]);
plan.layer{2}.cpu.vars.W = Wapprox;

% Forward prop
% Get error
error = 0;
plan.input.step = 1;
for i = 1:8
    plan.input.GetImage(0);
    ForwardPass(plan.input); 
    error = error + plan.classifier.GetScore(5);
    fprintf('%d / %d\n', error, i * plan.input.batch_size);
end

plan.layer{2}.cpu.vars.W = Worig;
