function init(path)
    global root_path;
    if (exist('root_path') ~= 1 || isempty(root_path))
        root_path = path;
        addpath(genpath(path));
    end
    
end