function ret = Val(var, name, val)
try
    ret = eval(sprintf('var.%s', name));
catch
    ret = val;
end
end