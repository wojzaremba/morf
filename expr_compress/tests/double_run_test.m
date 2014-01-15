clear all
global debug
debug = 2;
init();
load_mock_model();
S = MockScheduler(struct('max_errors', 200));
approx = MockApprox('_test', struct('A', {3, 4, 6, 8}), ...
                     struct('B', {2, 3}));                                 
S.Add(approx);
S.approx_logs{1}.ClearLog();
S.Run();
assert(S.compilation_counter == 5);
assert(approx.nr_execution == 5);
S.Printf();

fprintf('Next execution\n');
S = MockScheduler(struct('max_errors', 128));
approx = MockApprox('_test', struct('A', {3, 4}), ...
                     struct('B', {2}));   

S.Add(approx);
S.Printf();
S.Run();
assert(S.compilation_counter == 0);
assert(approx.nr_execution == 1);