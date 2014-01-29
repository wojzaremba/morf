function init()
    global root_path;
    if (exist('root_path', 'var') ~= 1 || isempty(root_path))
        root_path = sprintf('%s/', pwd);
        addpath(genpath(root_path));
    end
    
end
