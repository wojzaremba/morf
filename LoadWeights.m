function path = LoadWeights()
    cluster = is_cluster();
    if (~cluster)
        path = '/Users/Denton/Documents/morf-emily/expr_compress/monochromaticfilters/imagenet_weights.mat';
    elseif (cluster)
        path = '/misc/vlgscratch2/LecunGroup/bruna/all_imagenet_weights.mat';
    end
end