% 1. Run with approximation of CPU, and check that error is within range
% 2. Run with approxiamtion on GPU, (the same as above)
% 3. Prepare code for Misha.



% XXX : Extend number of test images (use other impl. of input layer).
% XXX : Functions to inspect co8ntent of this map of maps. 
% XXX : Deal with passing between C_ and Capprox_gen.

global plan debug;
debug = 2;
clearvars -except root_path plan;

init();
load_imagenet_model();


S = Scheduler(struct('max_errors', 110, 'no_compilation', 0));
S.Add(MonochromaticInput('', struct('num_image_colors', {4}), ...
                         struct('B_X', {32, 32}, ...
                                'B_Y', {4, 6}, ...
                                'imgsPerThread', {4, 4}, ...
                                'filtersPerThread', {6, 4}, ...
                                'colorsPerBlock', {1, 1}, ...
                                'scale', {0, 0}, ...
                                'checkImgBounds', {0, 0})));
S.approx_logs{1}.ClearLog();
S.Run();
S.Printf();

S.VerifyCompiled();