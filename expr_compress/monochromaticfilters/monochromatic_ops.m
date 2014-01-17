function [ops, mono_ops] = monochromatic_ops(N, K, stride, padding, F, C)
    M = (N + 2*padding) / stride;
    ops = M^2 * K^2 * 3 * F;
    mono_ops = M^2 * K^2 * F + N^2 * 3 * C;
end