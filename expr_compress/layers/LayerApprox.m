classdef LayerApprox < Layer
    properties
    end
    
    methods
        
        function obj = LayerApprox(json)
            obj@Layer(json);
        end     

        function Update(layer)
            global plan;
            lr = plan.lr;
            if (lr == 0)
                return;
            end
            momentum = plan.momentum;
            if (~layer.on_gpu)
                f = fields(layer.cpu.dvars);
                for i = 1:length(f)
                    if (strcmp(f{i}, 'X')) || (strcmp(f{i}, 'out'))
                        continue;
                    end
                    name = f{i};                 
                    eval(sprintf('layer.cpu.accum.%s = (1 - momentum) * layer.cpu.dvars.%s / plan.input.batch_size + momentum * layer.cpu.accum.%s;', name, name, name));
                    eval(sprintf('layer.cpu.vars.%s = layer.cpu.vars.%s - lr * layer.cpu.accum.%s;', name, name, name));
                end
            else                
                f = fields(layer.gpu.dvars);
                for i = 1:length(f)
                    if (strcmp(f{i}, 'X')) || (strcmp(f{i}, 'out'))
                        continue;
                    end
                    name = f{i};
                    vars_gid = eval(sprintf('layer.gpu.vars.%s', name));
                    dvars_gid = eval(sprintf('layer.gpu.dvars.%s', name));
                    accum_gid = eval(sprintf('layer.gpu.accum.%s', name));
                    
                    Capprox_gen(Scale, accum_gid, single(momentum), accum_gid);
                    Capprox_gen(Scale, dvars_gid, single((1 - momentum) * 1 / plan.input.batch_size), dvars_gid);
                    Capprox_gen(Add, accum_gid, dvars_gid, accum_gid);
                    Capprox_gen(Scale, dvars_gid, single(1 / (1 - momentum) * plan.input.batch_size), dvars_gid); % XXXXXXX : Create a single GPU function for this update.
                    Capprox_gen(Scale, accum_gid, single(lr), accum_gid); % XXX : Fix it (lose of numerical precision.
                    Capprox_gen(Subtract, vars_gid, accum_gid, vars_gid);
                    Capprox_gen(Scale, accum_gid, single(1 / lr), accum_gid); % XXX : Fix it (lose of numerical precision.
                end                
            end
        end
        
        function AddParamsOnlyGPU(obj, name, dims, includeDer)
            global plan
            if (obj.on_gpu == 1)
                vartype = {'vars', 'dvars', 'accum'};
                for i = 1 : length(vartype)
                    try
                        var = eval(sprintf('single(obj.cpu.%s.%s)', vartype{i}, name));
                        eval(sprintf('obj.gpu.%s.%s = plan.GetGID();', vartype{i}, name));
                        gid = eval(sprintf('obj.gpu.%s.%s', vartype{i}, name));                
                        Capprox_gen(CopyToGPU, gid, var);
                        plan.stats.total_vars_gpu = plan.stats.total_vars_gpu + prod(dims);                    
                    catch
                        % This variable don't have corresponding derivates.
                    end
                end
            end
        end
        
        % Establishes correspondence between layer outputs and derivatives of outputs on GPU.
        function Finalize(obj)
            global plan
            obj.InitWeights();
            dims = [plan.input.batch_size, obj.dims];
            obj.AddParamsOnlyCPU('out', dims, true);
            if (obj.layer_nr > 1)                
                pobj = plan.layer{obj.layer_nr - 1};
                obj.AddParamsOnlyCPU('X', [plan.input.batch_size, pobj.dims], true);                
            end
            if (obj.on_gpu)                
                % vars.out corresponds to the next layer vars.X.
                % dvars.X corrensponds to the previous layer dvars.out.
                obj.gpu.vars.out = plan.GetGID();
                Capprox_gen(CopyToGPU, obj.gpu.vars.out, single(obj.cpu.vars.out));
                plan.stats.total_vars_gpu = plan.stats.total_vars_gpu + prod(dims);               
                
                if (obj.layer_nr > 1)
                    obj.gpu.dvars.X = plan.GetGID();
                    Capprox_gen(CopyToGPU, obj.gpu.dvars.X, single(obj.cpu.dvars.X));
                    plan.stats.total_vars_gpu = plan.stats.total_vars_gpu + prod(dims);                   
                    pobj = plan.layer{obj.layer_nr - 1};
                    if (pobj.on_gpu)
                        obj.gpu.vars.X = pobj.gpu.vars.out;
                        pobj.gpu.dvars.out = obj.gpu.vars.X;
                    else
                        obj.gpu.vars.X = plan.GetGID();
                        Capprox_gen(CopyToGPU, obj.gpu.vars.X, single(obj.cpu.vars.X));
                        plan.stats.total_vars_gpu = plan.stats.total_vars_gpu + prod(dims);                                                
                    end
                end
            else
                if (obj.layer_nr > 1)       
                    pobj = plan.layer{obj.layer_nr - 1};
                    if (pobj.on_gpu)
                        pobj.gpu.dvars.out = plan.GetGID();
                        Capprox_gen(CopyToGPU, pobj.gpu.dvars.out, single(pobj.cpu.dvars.out));
                        plan.stats.total_vars_gpu = plan.stats.total_vars_gpu + prod(dims);                
                    end
                end
            end
            obj.DisplayInfo();
        end        
        
        function Upload(obj, name, val)
            if (obj.on_gpu == 1)
                gid = eval(sprintf('obj.gpu.vars.%s', name));
                Capprox_gen(CopyToGPU, gid, eval(sprintf('single(val)')));               
            end            
            eval(sprintf('obj.cpu.vars.%s = val;', name));            
        end

        function ret = Download(obj, name)
            gid = eval(sprintf('obj.gids.%s', name));
            ret = Capprox_gen(CopyFromGPU, gid);
        end
        
    end
end
