function ret = struct2str(struc)
    ret = strrep(evalc(['disp(struc)']), char(10), '_');
end