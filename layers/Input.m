classdef Input < Layer
    properties
        repeat
        batch_size
        translate
        step
        max_repeat         
    end
    methods
        function obj = Input(json)
            obj@Layer(json);
            global plan            
            obj.repeat = 1;
            obj.step = 1;            
            obj.batch_size = Val(json, 'batch_size', 6);            
            obj.translate = Val(json, 'translate', 0);            
            obj.max_repeat = 10000;    
            obj.gpu.vars.out = -1;
            obj.gpu.vars.Y = -1;
            plan.input = obj;             
        end       

        function FP_(obj) 
        end        
        
        function FP(obj) 
        end
        
        function dparams = BP(~, ~)
            dparams = struct();
        end        

        function GetImage(obj, train)        
            global plan
            [X, Y, obj.step] = GetImage_(obj, obj.step, train);
            obj.cpu.vars.out = X;
            obj.cpu.vars.Y = Y;
            if (obj.gpu.vars.out == -1)
                obj.gpu.vars.out = plan.GetGID();
            end
            if (obj.gpu.vars.Y == -1)
                obj.gpu.vars.Y = plan.GetGID();
            end
            if (obj.on_gpu)
                C_(CopyToGPU, obj.gpu.vars.out, single(obj.cpu.vars.out(:, :)));
                C_(CopyToGPU, obj.gpu.vars.Y, single(obj.cpu.vars.Y(:, :)));                
            end
        end
    end
end


