function img = ExtractCentral(img, f_size)
    s1 = min(size(img, 1), size(img, 2));
    s2 = max(size(img, 1), size(img, 2));
    r1 = (ceil((s2 - s1 + 1) / 2)) : (floor((s2 + s1) / 2));
    r2 = 1:s1;
    if (size(img, 1) < size(img, 2))
        img = img(r2, r1, :);
    else
        img = img(r1, r2, :);
    end   
    img = imresize(img, [f_size, f_size]);    
    if (size(img, 3) == 1)
        img = repmat(img, [1, 1, 3]);
    end
end