% XXX : Check why FP is called too early. Put assert(0) in FP, so we will
% get trace.
% XXX : Fix matlab initial dir.

% XXX: Add verification that on_gpu = 1.
% XXX: Run just a single layer, for FP.

global plan debug;
debug = 2;
clearvars -except root_path plan;
init();
load_imagenet_model();


S = Scheduler(struct('max_errors', 25, 'no_compilation', 0));

approx = NoApproximation(struct('suffix', '', 'on_gpu', 1), ...
                         struct('opt1', 0), ...
                         struct('opt2', 0));
S.Add(approx);

approx = MonochromaticInput(struct('suffix', '', 'on_gpu', 0), ...
                         struct('num_image_colors', {4,  6,  8, 12}), ...
                         struct('origNumColors',    {3,  3,  3,  3,  3,  3,  3, 3, 3}, ...
                                'B_X',              {32, 32, 32, 32, 32, 32, 32, 32, 32}, ...
                                'B_Y',              {4,  6,  4,  3,  4,  4,  8,  8,  4}, ...
                                'imgsPerThread',    {4,  4,  4,  4,  4,  4,  4,  4,  4}, ...
                                'filtersPerThread', {6,  4,  3,  4,  4,  8,  4,  4,  8}, ...
                                'colorsPerBlock',   {2,  2,  1,  1,  1,  2,  2,  4,  4}, ...
                                'scale',            {0,  0,  0,  0,  0,  0,  0,  0,  0}, ...
                                'checkImgBounds',   {0,  0,  0,  0,  0,  0,  0,  0,  0}));    
                                 
S.Add(approx);
S.approx_logs{1}.ClearLog();
S.approx_logs{2}.ClearLog();
S.Run();
S.Printf();
