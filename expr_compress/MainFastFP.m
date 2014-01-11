% XXX : Extend number of test images (use other impl. of input layer).
% XXX : Functions to inspect content of this map of maps. 
% XXX : remove unnessesary funcs from Cmono.cu

global root_path;
global plan debug;
debug = 0;
clearvars -except root_path plan;
if (exist('root_path') ~= 1 || isempty(root_path))
    init('/Volumes/denton/Documents/morf/');
end
if (exist('plan') ~= 1 || isempty(plan))
    load_imagenet_model();
end

S = Scheduler(struct('acceptance', 0.99));

S.Add(MonochromaticInput(struct('numImgColors', {3, 4, 6}), ...
                         struct('B_X', {32, 32, 32}, ...
                                'B_Y', {4, 4, 8}, ...
                                'imgsPerThread', {4, 4, 4}, ...
                                'filtersPerThread', {8, 6, 1}, ...
                                'colorsPerBlock', {1, 1, 1}, ...
                                'scale', {0, 0, 0}, ...
                                'checkImgBounds', {0, 0, 0})));
S.Run();
S.Printf();



