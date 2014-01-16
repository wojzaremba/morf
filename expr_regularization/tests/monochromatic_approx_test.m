clear all;

W = randn(96, 11, 11, 3);
num_colors = 16;

% Approximate W
[Wapprox1, recon_error1]  = monochromatic_approx(W, num_colors);
fprintf('||W - Wapprox1|| / ||W|| = %f\n', recon_error1);

% approximate Wapprox1
[Wapprox2, recon_error2]  = monochromatic_approx(Wapprox1, num_colors);
fprintf('||Wapprox1 - Wapprox2|| / ||Wapprox1|| = %f\n', recon_error2);