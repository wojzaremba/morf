clear all;
global root_path debug plan executed_mock_fc
debug = 0;
init();

jsons = {};
jsons{1} = struct('batch_size', 128, 'rows', 224, 'cols', 224, 'depth', 3, 'number_of_classes', 10, 'type', 'TestInput');
jsons{2} = struct('local_2d_patch', struct('patch_rows', 11, 'patch_cols', 11, 'stride_rows', 4, 'stride_cols', 4), ...
                   'on_gpu', 0, 'depth', 96, 'function', 'LINEAR', 'type', 'Conv');
jsons{3} = struct('function', 'LINEAR', 'rows', 1, 'cols', 1, 'depth', 10, 'type', 'FC', 'fully_connected', true);      
jsons{4} = struct('type', 'Softmax');
plan = Plan(jsons);    


S = MockScheduler(struct('max_errors', 110));
approx = NoApproximation(struct('suffix', '_test'), ...
                         struct('json', { jsons{2} }), struct());                                 
S.Add(approx);
S.approx_logs{1}.ClearLog();
S.Run();
S.Printf();

% XXX: Put here assertions.