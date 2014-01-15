clear all
global debug plan
debug = 2;
init();

dims = [96, 4, 4, 3];

jsons = {};
jsons{1} = struct('batch_size', 128, 'rows', 8, 'cols', 8, 'depth', 3, 'number_of_classes', 10, 'type', 'TestInput');
jsons{2} = struct('local_2d_patch', struct('patch_rows', 4, 'patch_cols', 4), ...
                  'depth', 96, 'function', 'LINEAR', 'type', 'Conv');
jsons{3} = struct('function', 'LINEAR', 'rows', 1, 'cols', 1, 'depth', 10, 'type', 'FC', 'fully_connected', true);      
jsons{4} = struct('type', 'Softmax');
plan = Plan(jsons);    

num_image_colors = 16;
colors = randn([num_image_colors, dims(4)]);
dec = randn(dims(1), dims(4));
Wmono = randn([dims(1), dims(2), dims(3)]);
S = randn(dims(1), dims(2)*dims(3));
assignment = reshape(repmat(1:num_image_colors', [1, dims(1) / num_image_colors]), dims(1), 1);
W = MonochromaticInput.ReconstructW(colors, dec, S, assignment, [dims(1), dims(4), dims(2), dims(3)]);
plan.layer{2}.cpu.vars.W = W;


S = Scheduler(struct('max_errors', 110, 'no_compilation', 0));
approx = MonochromaticInput('_test',  struct('num_image_colors', {num_image_colors}), ...
                            struct('B_X', {32, 32}, ...
                                   'B_Y', {4, 6}, ...
                                   'imgsPerThread', {4, 4}, ...
                                   'filtersPerThread', {6, 4}, ...
                                   'colorsPerBlock', {1, 1}, ...
                                   'scale', {0, 0}, ...
                                   'checkImgBounds', {0, 0}));                                 
S.Add(approx);
S.approx_logs{1}.ClearLog();
S.Printf();
S.Run();
