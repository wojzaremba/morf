classdef Scheduler < handle       
    properties
        approxs
        approx_logs
        acceptance
        path
    end    
    
    methods
        function obj = Scheduler(opt)
            obj.approxs = {};
            obj.approx_logs = {};
            obj.acceptance = opt.acceptance;
            obj.acceptance = 21; % XXX (it should be relative).
        end
        
        function Add(obj, approx)
            obj.approxs{end + 1} = approx;
            obj.approx_logs{end + 1} = ApproxLog(approx.name);
        end
        
        function Printf(obj)
           for i = 1:length(obj.approx_logs)
               fprintf('Printing approx_log for %s\n', obj.approxs{i}.name);
               obj.approx_logs{i}.Printf(); 
           end
        end
        
        function Run(obj)
            for i = 1 : length(obj.approxs)
                approx = obj.approxs{i};
                log = obj.approx_logs{i};
                fprintf('Executing approx : %s\n', approx.name);
                approx.ResetApproxVarsIter();                
                success = true;
                % Iterate over approximation parameters
                while (success)
                    [success, approx_vars] = approx.GetApproxVars(); % struct('numImgColors', 4])
                    Wapprox = [];  
                    
                    approx.ResetCudaVarsIter(); 
                    success2 = true;
                    % Iterate over cuda variables
                    while(success2)
                        [success2, cuda_vars] = approx.GetCudaVars(approx_vars);
                        if (log.IsProcessed(approx_vars, cuda_vars))
                            continue;
                        end 
                        
                        % Skip if not a valid combination of approximation
                        % and cuda variables
                        if (~approx.VerifyCombination(approx_vars, cuda_vars))
                            continue;
                        end
                        
                        % If we haven't computed the approximation yet, do
                        % it here
                        if (isempty(Wapprox))
                            [Wapprox, approx_ret] = approx.Approx(approx_vars);
                            test_error = obj.TestApprox(Wapprox);
                            log.SaveApproxInfo(approx_vars, struct('test_error', test_error));                
                        end
                        test_error = log.GetApproxInfo(approx_vars).test_error;
                        
                        if (test_error > obj.acceptance)                            
                            continue;
                        end
                        % Compile, filling in template variables with
                        % cuda_vars
                        obj.Compile(approx.name, approx_ret, cuda_vars);
                        
                        % Run the cuda code
                        cuda_results = approx.RunCuda(approx_ret); % XXX : Ideally unified
                        
                        % Log the results specific to these approx_vars and
                        % cuda_vars
                        log.AddCudaExecutionResults(approx_vars, cuda_vars, cuda_results);
                    end                    
                end
            end
        end
        
        % func_name is a a string, args and cuda_vars are structs
        function Compile(obj, func_name, args, cuda_vars)
            global root_path;
            % replace #mock with, func_name 
            fid_read = fopen(strcat(root_path, 'expr_compress/cuda/src/Capprox_template.cu'), 'r');
            fid_write = fopen(strcat(root_path, 'expr_compress/cuda/src/Capprox_gen.cu'), 'wt');
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

            fclose(fid_read);
            fclose(fid_write);
            % compile
            cd(strcat(root_path, '/expr_compress/cuda'));
            status = system('make mexapprox');
            cd(root_path);
            
            if status
                fprintf('Error compiling with target mexapprox\n');
%                 fid_read = fopen(args.cuda_true, 'r');
%                 line = fgets(fid_read);
%                 while ischar(line)
%                     disp(line);
%                     line = fgets(fid_read);
%                 end
%                 fclose(fid_read);
                assert(0);
            end
        end
        
        function test_error = TestApprox(obj, Wapprox)
            test_error = 0;
            % XXX
        end
        
        
        
    end    
end    
