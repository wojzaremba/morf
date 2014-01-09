function display_scheduler_maps(approxs, maps)
    for k = 1 : length(approxs)
        fprintf('Approx = %s\n', func2str(approxs{k}));
        map = maps{k};
        key_set = keys(map);
        for i = 1 : length(key_set)
            key = key_set{i};
            fprintf('\t%s ', key);
            val = map(key);
            test_error = val.test_error;
            fprintf('test_error = %f\n', test_error);
            if (~isfield(val, 'cuda_params_map'))
                continue;
            end
            assert(0);
        end
    end
end