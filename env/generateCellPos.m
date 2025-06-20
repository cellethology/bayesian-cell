function [circles]=generateCellPos(xycirc,rad)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function for generating tissue cells   %   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nCircles = xycirc(1) * xycirc(2);
circles = zeros(nCircles ,2);
r = rad;

xrange = linspace(0+r,200-r,xycirc(1)); %200-r
yrange = linspace(-400+r,400-r,xycirc(2));
nCircles = length(xrange)*length(yrange);
 
% angles = randi([0,2*pi],nCircles,1);
% circles = combvec(xrange,yrange)' + randn(nCircles,1).*[cos(angles),sin(angles)];
mu = [0,0];
Sigma = 400/r*eye(2);
circles = combvec(xrange,yrange)' + mvnrnd(mu,Sigma,nCircles);

end