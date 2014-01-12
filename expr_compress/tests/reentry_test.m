global root_path debug
debug = 2;
if (exist('root_path') ~= 1 || isempty(root_path))
    init('/Volumes/denton/Documents/morf/');
end

S = MockScheduler(struct('acceptance', 0.99));
approx = MockApprox('_reentry_test', ...
                     struct('A', {3, 4}), ...
                     struct('B', {2, 4}));                                  

S.Add(approx);
S.approx_logs{1}.ClearLog();
S.Run();
assert(approx.nr_execution == 2);

S = MockScheduler(struct('acceptance', 0.99));
approx = MockApprox('_reentry_test', ...
                     struct('A', {4, 6}), ...
                     struct('B', {2, 4}));                                  
S.Add(approx);
S.Run();
assert(approx.nr_execution == 3);