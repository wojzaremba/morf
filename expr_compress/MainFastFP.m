% XXX : Extend number of test images (use other impl. of input layer).
% XXX : Functions to inspect content of this map of maps. 
% XXX : remove unnessesary funcs from Cmono.cu

%clear all
% load_imagenet_model();
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
S.log.Printf();



