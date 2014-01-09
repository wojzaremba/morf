function MemoryCopyTest()
    Q1 = single(randn(10, 20));
    C_(CopyToGPU, 1, Q1);
    Q2 = single(randn(3, 20, 5));
    C_(CopyToGPU, 111, Q2);

    Q1 = Q1 + 10;
    X1 = PassVariable(1);
    Q2 = Q2 - 3;
    X2 = C_(CopyFromGPU, 111);


    assert(norm(X1(:) - Q1(:) + 10) < 1e-5);
    assert(norm(size(X1) - size(Q1)) == 0);
    assert(norm(X2(:) - Q2(:) - 3) < 1e-5);
end    
    
function X1 = PassVariable(param_nr)
    X1 = C_(CopyFromGPU, param_nr);
end