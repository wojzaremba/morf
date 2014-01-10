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