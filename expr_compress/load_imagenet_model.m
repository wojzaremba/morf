function load_imagenet_model()
    global plan root_path
    if (exist('plan') ~= 1 || isempty(plan) || length(plan.layer) < 10)

        json = ParseJSON('plans/imagenet.txt');
        json{1}.batch_size = 128;
        Plan(json, strcat(root_path, 'trained/imagenet'));
        plan.training = 0;
        plan.input.step = 1;
        plan.input.GetImage(0);
    end
end
