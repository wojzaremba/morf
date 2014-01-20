function [assignment, centers] = mykmeans(X, k, W, Wm)
% X : N x D matrix of data points
% k : number of clusters
% W : original weights we want to approximate
% Wm : 'monochromatic' weights used to reconstruct
    
    max_iter = 1000;
    
    N = size(X, 1);
    D = size(X, 2);

    % Initialize cluster centers as randomly chosen data point.
    r = randperm(N);
    r = r(1:k);
    centers = X(r, :);
    
    assignment = zeros(1, N);
    
    for iter = 1:max_iter
        % Assign each data point to a cluster
        for n = 1:N
            assignment(n) = assign_point(n, centers, W, Wm);
        end
        
        % Reassign cluster centers
        for c = 1:k
           centers(c, :) = get_center(X(assignment == c, :)); 
        end

    end

end

function center = get_center(X)
    center = mean(X, 1);
end

function assignment = assign_point(n, centers, W, Wm)
    assignment = -1;
    min_dist = 1e10;
    for c = 1:size(centers, 1)
        Wrec = reconstructW(centers(c, :), Wm(n, :));
        dist = norm(squeeze(W(n, :, :)) - Wrec);
        if (dist < min_dist) 
            min_dist = dist;
            assignment = c;
        end
    end
end

function Wrec = reconstructW(center, Wm)
    Wrec = center' * Wm;
end