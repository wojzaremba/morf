A = randn(100, 100);
B = randn(9, 9);
fs = size(A, 1) + size(B, 1) - 1;
a = ifft2(fft2(A, fs, fs) .* fft2(B, fs, fs));
q = a(5:104, 5:104) - conv2(A, B, 'same');
norm(q(:))