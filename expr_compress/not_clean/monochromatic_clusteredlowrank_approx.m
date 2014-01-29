global plan root_path
init();
%type = 'alex';
%load_imagenet_model(type);
load_imagenet_model();
Worig = plan.layer{2}.cpu.vars.W;
W = plan.layer{2}.cpu.vars.W;
K = 7;

% figure1 = figure('Position', [0, 0, 1000, 1000]);
% axes1 = axes('Parent',figure1);
% view(axes1,[14.5 34]);
% grid(axes1,'on');
% hold(axes1,'all');
W = permute(W, [1, 4, 2, 3]);
approx0  =zeros(size(W));
for f = 1 : size(W,1)
    [u,s,v]=svd(squeeze(W(f,:,:)),0);
    C(f, :) = u(:, 1);
    S(f, :) = s(1, 1) * v(:, 1); %v(:,1);
    chunk = u(:, 1) * s(1, 1) * v(:, 1)';
    Wapprox0(f, :, :, :) = reshape(chunk, 1, size(W, 2), size(W, 3), size(W, 4));
end

 
[assignment,colors] = litekmeans(C', args.num_colors);
colors = colors';

Wapprox1 = zeros(size(W));
for f=1:size(W,1)
    chunk = (colors(f,:)') * (S(f,:));
    Wapprox1(f,:,:,:)=reshape(chunk,1,size(W,2),size(W,3),size(W,4));
end 

% Now cluster the mono weights
num_clusters = 8;
rank = 6;
Wmono = reshape(S,size(W,1),size(W,3),size(W,4));
 assignment = litekmeans(Wmono(:, :)', num_clusters);
lambda=0.001;
CMat = SparseCoefRecovery(double(Wmono(:, :)'), 0, 'Lasso', lambda);
CKSym = BuildAdjacency(CMat, 0);
[Grps] = SpectralClusteringEven(CKSym, num_clusters);
assignment= Grps(:, 2);
Wmonoapprox = zeros(size(Wmono));
for l = 1 : num_clusters
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
collage_imnet(Wapprox1);
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
