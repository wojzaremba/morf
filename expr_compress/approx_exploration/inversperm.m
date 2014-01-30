function out=inversperm(in)

L=length(in);
for l=1:L
out(in(l))=l;
end

