function load_imagenet_model(type)    
    global plan root_path
    if (~exist('type', 'var'))
        type = 'matthew';
    end        
    if (exist('plan') ~= 1 || isempty(plan) || length(plan.layer) < 10)
        json = ParseJSON(sprintf('plans/imagenet_%s.txt', type));
        json{1}.batch_size = 128;
        Plan(json, sprintf('~/imagenet_data/imagenet_%s', type));
        plan.training = 0;
        plan.input.step = 1;
        plan.input.GetImage(0);
        ForwardPass(plan.input);
    end
end
