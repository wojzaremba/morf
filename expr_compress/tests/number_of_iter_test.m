global root_path;

clearvars -except root_path;
if (exist('root_path') ~= 1 || isempty(root_path))
    init('/Volumes/denton/Documents/morf/');
end

S = Scheduler(struct('acceptance', 0.99));

approx1 = MockApprox(struct('A', {3, 4, 6}), ...
                     struct('B', {2, 3}));

approx2 = MockApprox(struct('A', {3, 4, 6, 11}), ...
                     struct('B', {2, 3, 11}));
                 
                 
S.Add(approx1);
S.Add(approx2);
S.Run();vim
S.Printf();

assert(approx1.nr_execution == 4);
assert(approx2.nr_execution == 5);