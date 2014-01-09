function SmallImagenetModel(layer_nr)
global plan
Plan('plans/imagenet.txt', 'trained/imagenet');
plan.training = 0;
plan.input.ReloadData(1);
for i = layer_nr:length(plan.layer)
    plan.layer(end) = [];
end
plan.layer{end}.dims = [plan.layer{end}.dims(1), 1, 1];
for i = (length(plan.layer) - 1):-1:1
    player = plan.layer{i + 1};
    pdims = player.dims;
    plan.layer{i}.dims = [plan.layer{i}.dims(1), (pdims(2) - 1) * player.stride(1) + player.patch(1), (pdims(3) - 1) * player.stride(2) + player.patch(2)];
    plan.layer{i}.padding = [0, 0];
end
plan.layer{end + 1} = NN(struct('type', 'NN', 'depth', 0));
plan.layer{end - 1}.next = {plan.layer{end}};
