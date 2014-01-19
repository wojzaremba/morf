clear all
test = load('~/morf/data/imagenet/test');
X = zeros(128, 224, 224, 3);
for i = 1:128
   X(i, :, :, :) = test.data{i}.X;
end
minX = double(squeeze(-min(X, [], 1)));
maxX = double(squeeze(255 - max(X, [], 1)));
meanX = (minX + maxX) / 2;

f = dir('~/val/');
X = zeros(128, 224, 224, 3);
Y = zeros(size(X, 1), 1);
idx = 1;
offset = 2000;
for i = 1 : length(f)
    fprintf('%d\n', i);
    if (f(i).name(end) ~= 'G')
        continue
    end
    if (idx <= (offset + size(X, 1))) && (idx > offset)            
        name = ['~/val/', f(i).name];
        assert(str2num(name(end - 12 : end - 5)) == idx)
        A = double(imread(name));
        A = ExtractCentral(A, 255);
        A = A(16:(end - 16), 16:(end - 16), :);
        X(idx - offset, :, :, :) = A;
    end
    idx = idx + 1;
    if (idx > (offset + size(X, 1)))
        break;
    end    
end
X = X - repmat(reshape(meanX, [1, size(meanX, 1), size(meanX, 2), size(meanX, 3)]), [size(X, 1), 1, 1, 1]);

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

for i = 1 : (size(Y, 1) + offset)
    if (i <= (offset + size(Y, 1))) && (i > offset)
        key = meta.synsets(str2num(tline)).WNID(2:end);
        fprintf('old Y = %d, synset = %s\n', i, key);         
        Y(i - offset) = map(key);
    end
    tline = fgetl(fid1);    
end
fclose(fid1);


data = {};
for i = 1:size(X, 1);
    data{i}.X = single(squeeze(X(i, :, :, :)));
    tmp = zeros(1000, 1);
    tmp(Y(i)) = 1;
    data{i}.Y = tmp;
end
assert(sum(Y == 0) == 0);
save('~/morf/data/imagenet/val', 'data')

X = [reshape(data{3}.X, [1, 3, 224, 224]); reshape(data{5}.X, [1, 3, 224, 224])];

imagesc(min(max((data{1}.X + 200) / 400, 0), 1))