function ret = struct2str(struc)
    ret = regexprep(evalc(['disp(struc)']), '[\n]*', '__');
    ret = regexprep(ret, '  +', ' ');
end