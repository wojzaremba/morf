global plan
addpath(genpath('../'));
SmallImagenetModel(12);

params2 = plan.layer{end - 3}.params;
params3 = plan.layer{end - 2}.params;

[oX, ~] = plan.input.GetImage(10, 1);
ForwardPass(oX(:, :, 1:plan.layer{1}.dims(2), 1:plan.layer{1}.dims(3)), plan.layer{2});
X = plan.layer{end - 3}.params.X;
out = {};
for i = 1:4
    out{i} = plan.layer{end - 4 + i}.params.X;
end

Plan('plans/imagenet_short.txt');
plan.layer{2}.params = params2;
plan.layer{3}.params = params3;
ForwardPass(X, plan.layer{2});

q = plan.layer{3}.params.X - out{2};
norm(q(:))

fprintf('test = %f\n', Test(0));
% Run();
