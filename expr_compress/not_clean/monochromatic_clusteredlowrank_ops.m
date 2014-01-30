N = 224; % Input map size.
M = 110; % Output map size.
K = 7; % Filter size.
Nout = 96; % Number of output maps.
Nin = 3; % number of input maps.
rank = 6; % Rank of approximation.
num_clusters = 8; % Number of clusters used to cluster output features.
cluster_sz = Nout / num_clusters;

%% Original:
orig.num_ops = M * M * K * K * Nin * Nout;
orig.weights = K * K * Nin * Nout;
orig.mem_accesses = N * N * Nin + orig.weights;


%% Approximated
approx.num_ops = N * N * Nin * Nout + K * K * M * M * Nout + K * K * rank;
approx.weights = Nout * Nin + num_clusters * (rank * K * K + cluster_sz * rank);
approx.mem_accesses = N * N * Nin + approx.weights;

fprintf('orig.num_ops / approx.num_ops = %f\n', orig.num_ops / approx.num_ops);
fprintf('orig.mem_accesses / approx.mem_accesses = %f\n', orig.mem_accesses / approx.mem_accesses);
fprintf('orig.weights / approx.weights = %f\n', orig.weights / approx.weights);