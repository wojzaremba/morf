global plan
addpath(genpath('../'));
SmallImagenetModel(12);

data = {};
for i = 1:10
    fprintf('i = %d\n', i);
    [oX, ~] = plan.input.GetImage(i, 1);
    ForwardPass(oX(:, :, 1:plan.layer{1}.dims(2), 1:plan.layer{1}.dims(3)), plan.layer{2});
    data{i} = struct('X', squeeze(plan.layer{end - 3}.params.X), 'Y', squeeze(plan.layer{end}.params.X));
end
save('../data/imagenet_regression/train', 'data');