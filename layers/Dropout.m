classdef Dropout < Layer
    properties
        p
    end
    
    methods
        function obj = Dropout(json)
            obj@Layer(FillDefault(json));
            obj.p = json.p;
            obj.Finalize();
        end
        
        function FP(obj)
            global plan;
            vars = obj.cpu.vars;
            out = zeros(size(vars.X));
            if (plan.training)
                input = plan.input;
                rand('seed', 100000 * input.repeat + input.step);
                ran = rand(size(vars.X));
                idx = logical(ran > obj.p);
                obj.cpu.vars.idx = idx;
                out(idx) = vars.X(idx);
            else
                out = vars.X * (1 - obj.p);
            end
            obj.cpu.vars.out = out;
        end
        
        function BP(obj)                
            global plan
            data = obj.cpu.dvars.out;
            if (plan.training)
                dX = zeros(size(obj.cpu.vars.X));
                dX(obj.cpu.vars.idx) = data(obj.cpu.vars.idx);
            else
                dX = data * (1 - obj.p);
            end      
            obj.cpu.dvars.X = dX;
        end
        
        function InitWeights(obj)
            obj.AddParam('out', [prod(obj.dims(1:2)), obj.depth()], false);             
        end        
        
    end
end

function json = FillDefault(json)
json.type = 'Dropout';
json.one2one = true;
end