classdef Scheduler < handle       
    properties
        approxs
        approx_logs
        path
        no_compilation
        max_errors
    end    
    
    methods
        function obj = Scheduler(opt)
            obj.approxs = {};
            obj.approx_logs = {};
            obj.max_errors = opt.max_errors;
            obj.no_compilation = Val(opt, 'no_compilation', 0);
        end
        
        function Add(obj, approx)
            obj.approxs{end + 1} = approx;
            obj.approx_logs{end + 1} = ApproxLog(approx.FullName(), ...
                struct('max_errors', obj.max_errors, 'no_compilation', obj.no_compilation));
        end
        
        function Printf(obj)
           for i = 1:length(obj.approx_logs)
               printf(0, 'Printing approx_log for %s\n', obj.approxs{i}.name);
               obj.approx_logs{i}.Printf(); 
           end
        end

        function VerifyCompiled(obj)
           for i = 1:length(obj.approx_logs)
               printf(2, 'VerifyCompiled approx_log for %s\n', obj.approxs{i}.name);
               obj.approx_logs{i}.VerifyCompiled(); 
           end
        end        
        
        
        
        function Run(obj)
            for i = 1 : length(obj.approxs)
                approx = obj.approxs{i};
                log = obj.approx_logs{i};
                fprintf('Executing approx : %s\n', approx.name);
                approx.ResetApproxVarsIter();                

                % Iterate over approximation parameters
                [success, approx_vars] = approx.GetApproxVars(); % struct('numImgColors', 4])
                while (success)
                    Wapprox = [];                      
                    approx.ResetCudaVarsIter(); 
                    % Iterate over cuda variables
                    success2 = true;
                    while(success2)
                        [success2, cuda_vars] = approx.GetCudaVars(approx_vars);   
                        if (~success2)
                            break;
                        end
                        if (log.IsProcessed(approx_vars, cuda_vars))
                            continue;
                        end 
                        
                        % If we haven't computed the approximation yet, do
                        % it here
                        if (isempty(Wapprox))
                            [Wapprox, approx_ret] = approx.ApproxGeneric(approx_vars);
                            [test_error, orig_time] = approx.RunOrigConv(Wapprox);
                            log.SaveApproxInfo(approx_vars, struct('test_error', test_error, 'orig_time', orig_time));                
                        end
                        test_error = log.GetApproxInfo(approx_vars).test_error;
                        
                        if (test_error > obj.max_errors)
                            continue;
                        end
                        % Compile, filling in template variables with
                        % cuda_vars
                        obj.Compile(approx.name, approx_ret, cuda_vars);
                        
                        % Run the cuda code
                        printf(2, 'Running modified conv with approx_params = %s, cuda_params = %s\n', struct2str(approx_vars), struct2str(cuda_vars));
                        [test_error_cuda, approx_time] = approx.RunModifConv(approx_ret);
                        assert(test_error_cuda == test_error);
                        cuda_results = struct('approx_time', approx_time);
                        % Log the results specific to these approx_vars and
                        % cuda_vars
                        log.AddCudaExecutionResults(approx_vars, cuda_vars, cuda_results);                        
                    end   
                    [success, approx_vars] = approx.GetApproxVars();
                end
            end
        end
        
        % func_name is a a string, args and cuda_vars are structs
        function Compile(obj, func_name, args, cuda_vars)
            if (obj.no_compilation)
                printf(0, 'We are not compiling anything !!!');
                return;
            end
            global root_path;
            % replace #mock with, func_name 
            fid_read = fopen(strcat(root_path, '/expr_compress/cuda/src/Capprox_template.cu'), 'r');
            fid_write = fopen(strcat(root_path, '/expr_compress/cuda/src/Capprox_gen.cu'), 'wt');
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
            fields = fieldnames(cuda_vars);
            fid_read = fopen(args.cuda_template, 'r');
            fid_write = fopen(args.cuda_true, 'wt');
            line = fgets(fid_read);
            while ischar(line)
                for f = 1:length(fields)
                   line = strrep(line, strcat('#', fields{f}), num2str(getfield(cuda_vars, fields{f}))); 
                end
                line = strrep(line, '\n', '\\n');
                line = strrep(line, '%', '%%');
                fprintf(fid_write, line);
                line = fgets(fid_read);
            end
            fprintf('Compiling %s with approx_params: %s, cuda_params: %s\n', func_name, struct2str(args.approx_params), struct2str(cuda_vars));

            fclose(fid_read);
            fclose(fid_write);
            % compile
            cd(strcat(root_path, '/expr_compress/cuda'));
            [status, cmdout] = system('make mexapprox');
            cd(root_path);
            
            if status
                fprintf('Error compiling with target mexapprox : \n%s\n', cmdout);
                assert(0);
            end
        end
    end    
end    
