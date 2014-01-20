function out=colorescale(in)

out=(in-min(in(:)))/(max(in(:))-min(in(:)));
