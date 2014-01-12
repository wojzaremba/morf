% XXX : Extend number of test images (use other impl. of input layer).
% XXX : Functions to inspect content of this map of maps. 

global root_path;
global plan debug;
debug = 2;
clearvars -except root_path plan;
if (exist('root_path') ~= 1 || isempty(root_path))
    init('/Volumes/denton/Documents/morf/');
end
if (exist('plan') ~= 1 || isempty(plan))
    load_imagenet_model();
end

S = Scheduler(struct('acceptance', 0.99, 'no_compilation', 1));
S.Add(MonochromaticInput('', struct('numImgColors', {4}), ...
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