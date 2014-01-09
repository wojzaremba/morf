function [D, alpha, diff] = Transform(type, jacobian, D, alpha, lr)
global plan

[u, ~, ~] = svd(jacobian);
u = u(:, 1:size(jacobian, 2));
dim1 = size(D, 2);
dim2 = size(D, 3);
if (strcmp(type, 'translation') == 1)
    move = zeros(2, 28, 28);
    alpha = alpha + 1;
    fprintf('alpha = %f, ', alpha);
    for x = 1:dim1
        for y = 1:dim2
            move(1, x, y) = 0;
            move(2, x, y) = alpha;
        end
    end
    match = move;
elseif (strcmp(type, 'rotation') == 1)
    rot = zeros(2, dim1, dim2);
    alpha = alpha + pi / 100;
    fprintf('alpha = %f, ', 360 * alpha / (2 * pi));
    for x = 1:dim1
        for y = 1:dim2
            rot(1, x, y) = (cos(alpha) * (x - 14) - sin(alpha) * (y - 14)) + 14 - x;
            rot(2, x, y) = (sin(alpha) * (x - 14) + cos(alpha) * (y - 14)) + 14 - y;
        end
    end
    match = rot;    
else 
    assert(0);
end
shift = zeros(2, 28, 28);
for x = 1:dim1
    for y = 1:dim2
        shift(1, x, y) = x;
        shift(2, x, y) = y;
    end
end

cvx_begin quiet
variables dD(2, 28, 28)
minimize( norm(D(:) - shift(:) - lr * dD(:) - match(:), 2));
subject to
u' * (D(:) - shift(:) - lr * dD(:)) == 0.
cvx_end
D = D - lr * dD;
diff =  norm(D(:) - shift(:) - match(:)) / norm(match(:));
fprintf('norm(D - match) = %f, normalized = %f \n', norm(D(:) - match(:)), diff());
end