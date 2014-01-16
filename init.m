function init()
    global root_path;
    if (exist('root_path') ~= 1 || isempty(root_path))
        root_path = strcat(pwd, '/');
        addpath(genpath(root_path));
    end
    
end