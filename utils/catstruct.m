function s = catstruct(s1, s2)
    fields = fieldnames(s2);
    s = s1;
    for f = 1:length(fields)
       field = fields{f};
       s = setfield(s, field, getfield(s2, field));       
    end
end