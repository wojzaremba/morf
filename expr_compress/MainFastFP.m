% 1. Run with approximation of CPU, and check that error is within range
% 2. Run with approxiamtion on GPU, (the same as above)
% 3. Prepare code for Misha.
% 4. Extend number of test images (use other impl. of input layer).

global root_path;
global plan debug;
debug = 2;
clearvars -except root_path plan;
init();
load_imagenet_model();


S = Scheduler(struct('acceptance', 0.8, 'orig_test_error', 20, 'no_compilation', 0));
approx = MonochromaticInput(struct('suffix', ''), ...
                         struct('num_image_colors', {4}), ...
                         struct('B_X', {32, 32}, ...
                                'B_Y', {4, 6}, ...
                                'imgsPerThread', {4, 4}, ...
                                'filtersPerThread', {6, 4}, ...
                                'colorsPerBlock', {1, 1}, ...
                                'scale', {0, 0}, ...
                                'checkImgBounds', {0, 0}));    
                                 
S.Add(approx);
S.approx_logs{1}.ClearLog();
S.Run();
S.Printf();
