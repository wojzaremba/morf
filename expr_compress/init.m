function init()
    cluster = is_cluster();
    if (cluster)
        addpath(genpath('~/Documents/morf/'));
    else
        addpath(genpath('/Volumes/denton/Documents/morf/'));
    end   
end

% XXX : Copied code.
function ret = is_cluster()
    [~, name] = system('hostname');
    if (strfind(name, 'dynapool'))
        fprintf('We are running locally\n');
        ret = false;
    elseif (strfind(name, 'cs.nyu.edu'))
        fprintf('We are on a cluster\n');
        ret = true;
    else
        assert(0);
    end
end