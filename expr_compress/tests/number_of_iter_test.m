global root_path debug
debug = 2;
if (exist('root_path') ~= 1 || isempty(root_path))
    init('/Volumes/denton/Documents/morf/');
end

S = MockScheduler(struct('acceptance', 0.99));

approx1 = MockApprox('_test1', struct('A', {3, 4}), ...
                     struct('B', {2}));

approx2 = MockApprox('_test2', struct('A', {3}), ...
                     struct('B', {2, 3}));
                 
                 
S.Add(approx1);
S.Add(approx2);
S.approx_logs{1}.ClearLog();
S.approx_logs{2}.ClearLog();
S.Run();
S.Printf();

assert(approx1.nr_execution == 1);
assert(approx2.nr_execution == 1);