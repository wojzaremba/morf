clc;
global root_path debug plan executed_conv_mock
debug = 2;
if (exist('root_path') ~= 1 || isempty(root_path))
    init('/Volumes/denton/Documents/morf/');
end
executed_conv_mock = 0;
jsons = {};
jsons{1} = struct('batch_size', 3, 'rows', 2, 'cols', 3, 'depth', 4, 'type', 'TestInput');
jsons{2} = struct('function', 'LINEAR', 'rows', 1, 'cols', 1, 'depth', 10, 'type', 'FC', 'fully_connected', true);    
jsons{3} = struct('type', 'Softmax');

plan = Plan(jsons);    

S = MockScheduler(struct('acceptance', 0.99, 'no_compilation', 1));
approx = MockApprox('_test', struct('A', {3, 4}), ...
                     struct('B', {2, 3}));                                 
S.Add(approx);
S.approx_logs{1}.ClearLog();
S.Printf();
S.Run();
assert(executed_conv_mock == 2);