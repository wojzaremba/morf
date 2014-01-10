function load_imagenet_model()
    global plan
    addpath(genpath('.'));
    json = ParseJSON('plans/imagenet.txt');
    json{1}.batch_size = 128;
    Plan(json, 'trained/imagenet');
    plan.training = 0;
    plan.input.step = 1;
    plan.input.GetImage(0);
end
