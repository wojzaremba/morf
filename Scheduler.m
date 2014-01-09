classdef Scheduler < handle       
    properties
        funcs
        funcs_params
        funcs_cuda_params
        params_map
        acceptance
        path
    end    
    
    methods
        function obj = Scheduler(opt)
            obj.funcs = {};
            obj.funcs_params = {};
            obj.funcs_cuda_params = {};
            obj.params_map = {};
            obj.acceptance = opt.acceptance;
            obj.path = opt.path;
        end
        
        function Add(obj, func, params, cuda_params)
            obj.funcs{end + 1} = func;
            obj.funcs_params{end + 1} = params;
            obj.funcs_cuda_params{end + 1} = cuda_params;
            obj.params_map{end + 1} = containers.Map('KeyType', 'char', 'ValueType', 'any');
        end
        
        function Run(obj)
            for i = 1 : length(obj.funcs)
                func = obj.funcs{i};
                fprintf('Executing approx : %s\n', func2str(func));
                params = obj.funcs_params{i};
                cuda_params = obj.funcs_cuda_params{i};
                for j = 1 : length(params)
                    key = struct2str(params{j});
                    if (obj.params_map{i}.isKey(key))
                        continue;
                    end
                    [Wapprox, args] = func(params{j});
                    test_error = obj.TestApprox(Wapprox);
                    obj.params_map{i}(key) = struct('test_error', test_error, 'args', args); % XXX : missing architecture description.
                    if (test_error > obj.acceptance)
                        fprintf('Too many test errors = %d\n', test_error);
                        continue;
                    end 
                    
                    % compile cuda code
                    %obj.Compile(func2str(func), args, cuda_params{j});
                    
                    % run cuda code
                    
                    
                    
                end
            end
        end
        
        % func_name is a a string, args and cuda_params are structs
        function Compile(obj, func_name, args, cuda_params)
            % replace #mock with, func_name 
            fid_read = fopen(strcat(obj.path, 'cuda/src/Capprox_template_.cu'), 'r');
            fid_write = fopen(strcat(obj.path, 'cuda/src/Capprox_.cu'), 'wt');
            line = fgets(fid_read);
            while ischar(line)
                line = strrep(line, '#mock' , func_name); 
                line = strrep(line, '\n', '\\n');
                line = strrep(line, '%', '%%');
                line = strrep(line, '\', '\\');
                fprintf(fid_write, line);
                line = fgets(fid_read);
            end

            fclose(fid_read);
            fclose(fid_write);
            
            % replace #var with var in func_params.template_file ->
            % func_params.true_file
            fields = fieldnames(cuda_params);
            fid_read = fopen(args.cuda_template, 'r');
            fid_write = fopen(args.cuda_true, 'wt');
            line = fgets(fid_read);
            while ischar(line)
                for f = 1:length(fields)
                   line = strrep(line, strcat('#', fields{f}), num2str(getfield(cuda_params, fields{f}))); 
                end
                line = strrep(line, '\n', '\\n');
                line = strrep(line, '%', '%%');
                fprintf(fid_write, line);
                line = fgets(fid_read);
            end

            fclose(fid_read);
            fclose(fid_write);
            
            
            % compile
            cd(args.compile_path);
            status = system('make mexapprox');
            
            if status
                fprintf('Error compiling with target mexapprox\n');
            end
        end
        
        function RunCuda(obj)
            
        end
        
        function test_error = TestApprox(obj, Wapprox)
            test_error = 0;
        end
        
        
        
    end    
end    
