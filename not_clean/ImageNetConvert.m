% clear all
global plan
addpath(genpath('../'));
q = load('data/imagenet/imagenet-weight');
Plan('plans/imagenet.txt');

plan.input.train = [];
plan.input.test = [];
for i = 1:length(plan.layer)
    try 
        plan.layer{i}.cpu = struct();
        plan.layer{i}.gpu = struct();        
    catch
    end
end
plan.gid = 0;

plan.layer{2}.cpu.vars.W = reshape(q.conv1_weight', [96, 11, 11, 3]);
plan.layer{2}.cpu.vars.B = q.conv1_bias;

plan.layer{5}.cpu.vars.W = reshape(q.conv2_weight', [256, 5, 5, 96]);
plan.layer{5}.cpu.vars.B = q.conv2_bias;

plan.layer{8}.cpu.vars.W = reshape(q.conv3_weight', [384, 3, 3, 256]);
plan.layer{8}.cpu.vars.B = q.conv3_bias;

plan.layer{9}.cpu.vars.W = reshape(q.conv4_weight', [384, 3, 3, 384]);
plan.layer{9}.cpu.vars.B = q.conv4_bias;

plan.layer{10}.cpu.vars.W = reshape(q.conv5_weight', [256, 3, 3, 384]);
plan.layer{10}.cpu.vars.B = q.conv5_bias;

plan.layer{12}.cpu.vars.W = q.fc6_weight';
plan.layer{12}.cpu.vars.B = q.fc6_bias';

plan.layer{13}.cpu.vars.W = q.fc7_weight';
plan.layer{13}.cpu.vars.B = q.fc7_bias';

plan.layer{14}.cpu.vars.W = q.fc8_weight';
plan.layer{14}.cpu.vars.B = q.fc8_bias';

save('../trained/imagenet', 'plan', '-v7.3');