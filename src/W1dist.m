function [W1dist,grad] = W1dist(x,y)

x = x(:);
y = y(:);

m = length(y);
xcdf = cumsum(x/sum(x));
ycdf = cumsum(y/sum(y));
phi = xcdf-ycdf;

phi = phi - median(phi);
sgnPhi = sign(phi);
W1dist = sum(abs(phi));

if nargout > 1 % gradient
    grad = ((tril(ones(m))-xcdf)'*sgnPhi)/sum(x);
end

end

