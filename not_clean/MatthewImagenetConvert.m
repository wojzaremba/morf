clear all
global plan
w = load('~/morf/trained/epoch0000_phase1');

addpath(genpath('../'));
Plan('plans/imagenet_matthew.txt');

plan.input.train = [];
plan.input.test = [];
convs = [2, 5, 8, 9, 10];
fulls = [12, 14, 16];
for i = 1:length(plan.layer)
    try 
        if (sum(convs == i) > 0) || (sum(fulls == i) > 0)
            continue;
        end
        plan.layer{i}.cpu = struct();
        plan.layer{i}.gpu = struct();        
    catch
    end
end
plan.gid = 0;

for i = 1:length(convs)
    fprintf('Setting up %dth conv\n', i);
    W = permute(w.params.F{i}, [4, 1, 2, 3]);
    B = w.params.forward_conv_b{i}';
    assert(sum(size(W) ~= size(plan.layer{convs(i)}.cpu.vars.W)) == 0);
    assert(sum(size(B) ~= size(plan.layer{convs(i)}.cpu.vars.B)) == 0);
    plan.layer{convs(i)}.cpu = struct();
    plan.layer{convs(i)}.gpu = struct();        
    plan.layer{convs(i)}.cpu.vars.W = W;
    plan.layer{convs(i)}.cpu.vars.B = B;
end

for i = 1:length(fulls)
    fprintf('Setting up %dth full\n', i);
    W = w.params.W{i};
    B = w.params.b{i};
    assert(sum(size(W) ~= size(plan.layer{fulls(i)}.cpu.vars.W)) == 0);
    assert(sum(size(B) ~= size(plan.layer{fulls(i)}.cpu.vars.B)) == 0);    
    plan.layer{fulls(i)}.cpu = struct();
    plan.layer{fulls(i)}.gpu = struct();    
    plan.layer{fulls(i)}.cpu.vars.W = W;
    plan.layer{fulls(i)}.cpu.vars.B = B;
end
save('../trained/imagenet_matthew', 'plan', '-v7.3');