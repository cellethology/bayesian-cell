function [output] = ellipsePerimeter(a,b)
% approximate perimeter of ellipse with semi-axes a and b
% max 0.04% error everywhere

h = (a-b).^2/(a+b).^2;

output = pi.*(a + b)*(1+ 3*h/(10+sqrt(4-3*h)));

end

