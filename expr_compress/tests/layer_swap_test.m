clear all
global debug plan executed_mock_fc
debug = 2;
init();
load_mock_model();
executed_mock_fc = 0;
load_mock_model();

randn('seed', 1)
plan.layer{2}.cpu.vars.B = randn(size(plan.layer{2}.cpu.vars.B)) / 1000;
S = MockScheduler(struct('max_errors', 190, 'no_compilation', 1));
approx = MockApprox('_test', struct('A', {3, 4}), ...
                     struct('B', {2, 3}));                                 
S.Add(approx);
S.approx_logs{1}.ClearLog();
S.Printf();
S.Run();
assert(executed_mock_fc == 2);