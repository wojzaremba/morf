% clear all;
global plan
addpath(genpath('../'));
name = 'mnist_conv';
Plan(['plans/', name '.txt'], ['trained/', name]);
plan.training = 0;

external_tools = '/Users/wojto/Dropbox/external/';
addpath([external_tools, 'mosek/7/toolbox/r2013a']);
setenv('MOSEKLM_LICENSE_FILE', [external_tools, 'mosek/mosek.lic']);
setenv('LD_LIBRARY_PATH', [external_tools, 'mosek/7/tools/platform/osx64x86/bin']);

input = plan.input;
input.ReloadData(1);
X_ = input.GetImage(1, 0);

X1 = zeros(28, 28, 11, 11);
for x = -5:5
    for y = -5:5
        X1(max(1 + x, 1):min(28 + x, 28), max(1 + y, 1):min(28 + y, 28), x + 6, y + 6) = X_(1, 1, max(1 - x, 1):min(28 - x, 28), max(1 - y, 1):min(28 - y, 28));
    end
end

X1 = X1(:, :, :);
X1 = permute(X1, [3, 1, 2]);
X1 = reshape(X1, [size(X1, 1), 1, size(X1, 2), size(X2, 3)]);
input.ReloadData(size(X1, 1));

ForwardPass(X1, plan.layer{2});



ydata = tsne(plan.layer{end - 1}.params.X);
scatter(ydata(:, 1), ydata(:, 2));