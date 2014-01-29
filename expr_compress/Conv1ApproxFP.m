global plan debug;
debug = 2;
clearvars -except root_path plan;
init();
load_imagenet_model();


approx_vars = struct('num_image_colors', 8);

approx = MonochromaticInput(struct('suffix', '_cpu', 'on_gpu', 0), ...
                            approx_vars, ...
                            struct('notused',    {3}));
                        
[Wapprox, ret_approx] = approx.ApproxGeneric(approx_vars);

% Do forward prop with swapped out layer
approx_time = approx.RunModifFP(ret_approx);
                        