function GC()
clc;
plan_json = ParseJSON('plans/tests.txt');
for i = 1 : length(plan_json)
    jsons = {};
    if (strcmp(plan_json{i}.type, 'Softmax'))
        jsons{end + 1} = struct('batch_size', 3, 'rows', 1, 'cols', 1, 'depth', 4, 'type', 'TestInput');
    else
        jsons{end + 1} = struct('batch_size', 3, 'rows', 8, 'cols', 10, 'depth', 4, 'type', 'TestInput');
    end
    jsons{end + 1} = plan_json{i};
    plan = Plan(jsons);
    plan.layer{2}.bp_dx_on_first = 1;
    fprintf('\n\nVerifing %d layer %s\n', i, plan.layer{end}.type);
    if (~VerifyLayer())
        assert(0);
        return;
    end
end
end

function passed = VerifyLayer()
global plan;
passed = true;
layer = plan.layer{end};
dims = layer.dims;
h = 1e-9;
eps = 1e-4;
plan.input.GetImage(1);
vars = layer.cpu.vars;
vars.X = plan.input.cpu.vars.out;
back_in = randn([size(vars.X, 1), dims]);
if (strcmp(layer.type, 'Softmax'))
    back_in(:) = 1;
end
layer.cpu.vars = vars;
layer.FP();
if (~ismethod(layer, 'BP'))
    return;
end
layer.cpu.dvars.out = back_in;
layer.BP();
dvars = layer.cpu.dvars;
f = fields(dvars);
for i = 1:length(f)
    if (strcmp(f{i}, 'out'))
        continue;
    end
    fprintf('Trying verify %s derivative....', f{i});
    name = f{i};
    V = eval(sprintf('layer.cpu.vars.%s', name));
    dV = eval(sprintf('layer.cpu.dvars.%s', name));
    dV_num = zeros(size(dV));
    for pos = 1:length(V(:))
        fprintf('.');
        Vcopy_a = V;
        Vcopy_a(pos) = Vcopy_a(pos) - h;
        eval(sprintf('layer.cpu.vars.%s = Vcopy_a;', name));
        layer.FP();
        out_a = layer.cpu.vars.out;
        
        Vcopy_b = V;
        Vcopy_b(pos) = Vcopy_b(pos) + h;
        eval(sprintf('layer.cpu.vars.%s = Vcopy_b;', name));
        layer.FP();
        out_b = layer.cpu.vars.out;
        dV_num(pos) = dot((out_b(:) - out_a(:)) ./ (2 * h), back_in(:));
    end
    diff = dV_num - dV;
    try
        assert(norm(diff(:)) / max(norm(dV_num(:)), 1) < eps);
        assert(length(dV_num(:)) > 0)
        assert(length(dV(:)) > 0)
        fprintf('%s derivative check DONE', f{i});
        if (norm(dV_num(:)) < 1e-5)
            fprintf(', \nWARNING values close to zero (GC might be unreliable)\n');
            passed = false;
            return;
        end
        fprintf('\n');
    catch
        fprintf('\nnorm diff = %f\n', norm(diff(:)));
        dV_num ./ dV
        fprintf('%s derivative FAILED\n', f{i});
        passed = false;
        return;
    end
end
end
