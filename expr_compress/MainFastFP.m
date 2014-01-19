% 1. Run with approximation of CPU, and check that error is within range
% 2. Run with approxiamtion on GPU, (the same as above)
% 3. Prepare code for Misha.


% XXX : Extend number of test images (use other impl. of input layer).
% XXX : Functions to inspect co8ntent of this map of maps. 
% XXX : Deal with passing between C_ and Capprox_gen.
% 4. Extend number of test images (use other impl. of input layer).

global plan debug;
debug = 2;
clearvars -except root_path plan;
init();
load_imagenet_model();


S = Scheduler(struct('acceptance', 0.8, 'orig_test_error', 20, 'no_compilation', 0));
approx = MonochromaticInput(struct('suffix', '', 'on_gpu', 0), ...
                         struct('num_image_colors', {12, 16}), ...
                         struct('origNumColors', {3, 3, 3}, ...
                                'B_X', {32, 32, 32}, ...
                                'B_Y', {4, 6, 4}, ...
                                'imgsPerThread', {4, 4, 4}, ...
                                'filtersPerThread', {4, 4, 3}, ...
                                'colorsPerBlock', {2, 1, 2}, ...
                                'scale', {0, 0, 0}, ...
                                'checkImgBounds', {0, 0, 0}));    
                                 
S.Add(approx);
S.approx_logs{1}.ClearLog();
S.Run();
S.Printf();

% XXX : During real execution this should be uncommented.
%S.VerifyCompiled();
