clear all
global debug
debug = 2;
init();
load_mock_model();
S = MockScheduler(struct('max_errors', 128));
approx = MockApprox(struct('suffix', '_reentry_test'), ...
                    struct('A', {3, 4}), ...
                    struct('B', {2, 4}));                                  

S.Add(approx);
S.approx_logs{1}.ClearLog();
S.Run();
assert(approx.nr_execution == 2);

S = MockScheduler(struct('max_errors', 128));
approx = MockApprox(struct('suffix', '_reentry_test'), ...
                    struct('A', {4, 6}), ...
                    struct('B', {2, 4}));                                  
S.Add(approx);
S.Run();
assert(approx.nr_execution == 3);
