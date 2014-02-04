function [minassignment, mincenters] = mykmeans(X, k, W, Wm)
% X : N x D matrix of data points
% k : number of clusters
% W : original weights we want to approximate
% Wm : 'monochromatic' weights used to reconstruct
    
    max_iter = 2000;
    
    N = size(X, 1);
    D = size(X, 2);

    % Initialize cluster centers as randomly chosen data point.
    mindist = Inf;
    minassignment = [];
    mincenters = [];
    rand('seed', 3);
    for rep = 1:100
        fprintf('rep = %d\n', rep);
        r = randperm(N);
        r = r(1:k);
        centers = X(r, :);

        assignment = zeros(1, N);

        old_assignment = 0;
        for iter = 1:max_iter
            % Assign each data point to a cluster
            for n = 1:N
                assignment(n) = assign_point(n, centers, W, Wm);
            end
            for f=1:size(W,1)
                chunk = (centers(assignment(f),:)') * (Wm(f,:));
                Wapprox(f,:,:,:) = reshape(chunk,1,size(W,2),size(W,3),size(W,4));
            end             
            dist = norm(Wapprox(:) - W(:)) / norm(W(:));
            fprintf('iter = %d, dist = %f \n', iter, dist);            
            
            if (dist < mindist)
                mindist = dist;
                minassignment = assignment;
                mincenters = centers;
            end                       
            
            % Reassign cluster centers
            for c = 1:k
               centers(c, :) = get_center(X(assignment == c, :)); 
            end
            if (norm(old_assignment(:) - assignment(:)) == 0) || (isnan(sum(old_assignment(:))))
                break;
            end
            old_assignment = assignment;    
        end
    end

end

function center = get_center(X)
    center = mean(X, 1);
end

function assignment = assign_point(n, centers, W, Wm)
    assignment = -1;
    min_dist = Inf;
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