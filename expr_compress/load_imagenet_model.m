function load_imagenet_model()
    global plan;
    addpath(genpath('.'));
    json = ParseJSON('plans/imagenet.txt');
    json{1}.batch_size = 128;
    if (~is_cluster())
    	Plan(json, '/Users/Denton/imagenet');
    else
    	Plan(json, 'trained/imagenet');
    end
    plan.training = 0;
    plan.input.step = 1;
    plan.input.GetImage(0);
end
