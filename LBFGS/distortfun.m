function [ c, g ] = distortfun(x)
    global plan
    orig_data = plan.input.cpu.vars.out;
    active_size = size(orig_data, 1);
    orig_size = size(orig_data);
    resize_size = [active_size orig_size(2:end)];
    x = reshape(x, resize_size);
    batch_size = size(orig_data, 1);
    plan.input.badX(plan.input.active_indices, :, :) = x;
    plan.input.cpu.vars.out = x;
    ForwardPass(plan.input);
    diff = x - plan.input.X(plan.input.active_indices, :, :);    
    distance_loss = sum(diff(:) .^ 2) / batch_size;
    classifier_cost = plan.classifier.Cost();
    c = classifier_cost + plan.input.lambda * distance_loss;
    for i = length(plan.layer) : -1 : 2
        layer = plan.layer{i};
        layer.BP();
        plan.layer{i - 1}.cpu.dvars.out = layer.cpu.dvars.X;
    end
    g = plan.layer{2}.cpu.dvars.X(:, :, :);
    g = g(:) + 2 * plan.input.lambda * diff(:) / batch_size;
end
