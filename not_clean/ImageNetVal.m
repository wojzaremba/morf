clear all
meanX = zeros(224, 224, 3);
Y = zeros(50000, 1);
for i = 1 : 50000
    fprintf('%d\n', i);
    name = sprintf('~/val/ILSVRC2012_val_%08d.JPEG', i);    
    img = imread(name);
    img = ExtractCentral(img, 255);
    img = img(16:(end - 16), 16:(end - 16), :);
    imwrite(img, sprintf('~/val_cropped/ILSVRC2012_val_%08d.JPEG', i));
    meanX = meanX + double(img);
end
meanX = single(meanX / 50000);


meta = load('~/morf/data/imagenet/meta.mat');
fid1 = fopen('~/morf/data/imagenet/labels.txt');
tline = fgetl(fid1);
map = containers.Map;
idx = 1;
while ischar(tline)   
    fprintf('Reading %s\n', tline);
    id = tline(end - 7:end);
    map(id) = idx;
    idx = idx + 1;
    tline = fgetl(fid1);      
end
fclose(fid1);

fid1 = fopen('~/morf/data/imagenet/ILSVRC2012_validation_ground_truth.txt');
tline = fgetl(fid1);

for i = 1 : 50000
    key = meta.synsets(str2num(tline)).WNID(2:end);
    fprintf('old Y = %d, synset = %s\n', i, key);         
    Y(i) = map(key);
    tline = fgetl(fid1);    
end
fclose(fid1);

save('~/val_cropped/meta', 'meanX', 'Y');