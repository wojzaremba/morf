clear all
global debug
debug = 2;
init();
load_mock_model();
S = MockScheduler(struct('max_errors', 128));

approx1 = MockApprox(struct('suffix', '_test1'), ...
                     struct('A', {3, 4}), ...
                     struct('B', {2}));

approx2 = MockApprox(struct('suffix', '_test2'), ...
                     struct('A', {3}), ...
                     struct('B', {2, 3}));
                                  
S.Add(approx1);
S.Add(approx2);
S.approx_logs{1}.ClearLog();
S.approx_logs{2}.ClearLog();
S.Run();
S.Printf();

assert(approx1.nr_execution == 1);
assert(approx2.nr_execution == 1);
