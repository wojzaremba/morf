function [jacobian, out] = GetJacobian(X)
    global plan
    fdim = prod(plan.layer(end).v.dims);
    assert(size(X, 1) == 1);
    plan.batch_size = fdim;    
    out = FP(repmat(X, [fdim, 1, 1, 1]));
    BP(eye(fdim));
    jacobian = plan.layer(2).v.scratch.jacobian';
end