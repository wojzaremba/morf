global plan root_path
init();
type = 'matthew';
load_imagenet_model(type);

W = plan.layer{5}.cpu.vars.W;