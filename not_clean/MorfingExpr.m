% XXXXX : Look into single position during deconvolution !!!.



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
X1 = input.GetImage(1, 0);
% X_ = input.GetImage(1, 0);
% X1 = zeros(1, 1, 28, 28);
% X1(1, 1, :, 1:21) = X_(1, 1, :, 8:end);
X2 = input.GetImage(2, 0);
fedge = input.dims(2);

ForwardPass(X2, plan.layer{2});
% act2 = plan.layer{2}.scratch.forward_act;
act2 = plan.layer{4}.params.X;

ForwardPass(X1, plan.layer{2});
% act1 = plan.layer{2}.scratch.forward_act;
act1 = plan.layer{4}.params.X;

colormap(gray);
% for a = 0:0.1:1  
a = 0;
fprintf('a = %f\n', a);
X = (1 - a) * act1 + a * act2;
[val, idx] = max(X(:));
X(:) = 0;
X(idx) = val;
% Xpool = plan.layer{2}.Invert(X);
Xout = plan.layer{2}.InvertPooling(X);


imagesc([squeeze(X1), squeeze(Xout), squeeze(X2)]);
%     pause;
% end

