function out=tensorize(f1,f2,f3,f4)

K=nargin;

S=zeros(K,1);

S(1) = length(f1);
S(2) = length(f2);
S(3) = length(f3);
if nargin > 3
S(4) = length(f4);
end

tmp = f1*f2';
tmp = tmp(:)*f3';
if nargin > 3
tmp = tmp(:)*f4';
out = reshape(tmp,S(1), S(2), S(3), S(4));
else
out = reshape(tmp,S(1), S(2), S(3));
end

end


