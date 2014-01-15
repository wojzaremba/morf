clear all
global root_path debug
debug = 2;
init();
load_mock_model();
S = MockScheduler(struct('acceptance', 0.99, 'orig_test_error', 128));
approx = MockApprox('_test', struct('A', {3, 4, 6, 8}), ...
                     struct('B', {2, 3}));                                 
S.Add(approx);
S.approx_logs{1}.ClearLog();
S.Run();
assert(S.compilation_counter == 5);
assert(approx.nr_execution == 5);
S.Printf();

fprintf('Next execution\n');
S = MockScheduler(struct('acceptance', 0.99, 'orig_test_error', 128));
approx = MockApprox('_test', struct('A', {3, 4}), ...
                     struct('B', {2}));   

S.Add(approx);
S.Run();
try
    S.VerifyCompiled();
    assert(0);
catch    
end
