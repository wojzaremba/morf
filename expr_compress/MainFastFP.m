% 1. Run with approximation of CPU, and check that error is within range
% 2. Run with approxiamtion on GPU, (the same as above)
% 3. Prepare code for Misha.



% XXX : Extend number of test images (use other impl. of input layer).
% XXX : Functions to inspect content of this map of maps. 
% XXX : investigate why multiplication by 2 isnt necessary in mock FP

global root_path;
global plan debug;
debug = 2;
clearvars -except root_path plan;

init('/Volumes/denton/Documents/morf/');
load_imagenet_model();


S = Scheduler(struct('acceptance', 0.99, 'no_compilation', 1));
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