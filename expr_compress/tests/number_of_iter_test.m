clear all
global root_path debug
debug = 2;
if (exist('root_path') ~= 1 || isempty(root_path))
    init('/Volumes/denton/Documents/morf/');
end

S = Scheduler(struct('acceptance', 0.99));

approx1 = MockApprox(struct('A', {3, 4}), ...
                     struct('B', {2}));

approx2 = MockApprox(struct('A', {3, 4}), ...
                     struct('B', {2, 3}));
                 
                 
S.Add(approx1);
% S.Add(approx2);
S.Run();
S.Printf();

assert(approx1.nr_execution == 1);
% assert(approx2.nr_execution == 2);