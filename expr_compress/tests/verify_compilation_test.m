clear all
global debug
debug = 2;
init();
load_mock_model();
S = MockScheduler(struct('max_errors', 128));
approx = MockApprox(struct('suffix', '_test'), ...
                    struct('A', {3, 4, 6, 8}), ...
                    struct('B', {2, 3}));                                 
S.Add(approx);
S.approx_logs{1}.ClearLog();
S.Run();
assert(S.compilation_counter == 5);
assert(approx.nr_execution == 5);
S.Printf();

fprintf('Next execution\n');
S = MockScheduler(struct('max_errors', 128));
approx = MockApprox(struct('suffix', '_test'), ...
                    struct('A', {3, 4}), ...
                    struct('B', {2}));   
S.Add(approx);
S.Run();
try
    S.VerifyCompiled();
    assert(0);
catch    
end
