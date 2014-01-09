function VerifyJacobian(dX, jacobian, node_data)
A = sparse(double(dX(1, :)));
B = squeeze(jacobian(:, :)) * sparse(double(squeeze(node_data(1, :))'));
assert(norm(A(:) - B(:)) < 1e-4);
end
