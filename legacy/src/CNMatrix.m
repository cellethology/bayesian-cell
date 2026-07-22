function [L R] = CNMatrix(param)
% Notes:
% 1) we will eventually solve (Left and Right sides):
%    Lw(t+1) = Rw(t)
%    where the vector w = [f1;...;fN;F_cyto]
% 2) build Crank-Nicholson matrix with L & R = [D u; v 1], where
%    D is param.N x param.N
%    u is 1 x param.N
%    v is param.N x 1
% 3) param.d is either a scalar (uniform diffusivity) or a length-N vector
%    giving the diffusivity at each membrane bin. The vector case discretises
%    the flux-conservative form d/dx( d(x) df/dx ) with d evaluated at bin
%    FACES, so the column sums of DL and DR telescope to 1/dt on the periodic
%    ring and sum(f) is conserved exactly -- the property the augmented row v
%    below relies on. For scalar d this reduces identically to the original
%    constant-coefficient stencil.

%1) useful constants
N = param.N;
d = param.d(:)';
if isscalar(d)
    d = d*ones(1,N);
elseif numel(d) ~= N
    error('CNMatrix:badDiffusivity', ...
          'param.d must be a scalar or a length-%d vector, got %d elements', ...
          N, numel(d))
end

% diffusivity at faces: dface(i) sits between bin i and bin i+1 (periodic)
dface = (d + circshift(d,-1))/2;
ap = dface/(2*param.dx^2);              % coupling from bin i to i+1
am = circshift(dface,1)/(2*param.dx^2); % coupling from bin i to i-1

%2) diagonal for Left (DL) and Right (DR) sides
% (am+ap) is grouped so that for uniform d it evaluates bit-identically to the
% original 1/dt + 2*alpha, keeping old runs exactly reproducible
DL = diag(1/param.dt + (am + ap)) - diag(am(2:end),-1) - diag(ap(1:end-1),1);
% assume circular space
DL(1,end) = -am(1);
DL(end,1) = -ap(end);

DR = diag(1/param.dt - (am + ap)) + diag(am(2:end),-1) + diag(ap(1:end-1),1);
% assume circular space
DR(1,end) = am(1);
DR(end,1) = ap(end);

%3) u,v
u = zeros(N,1); % u update in solver
v = ones(1,N+1);  %

%4)
L = [DL u; v];
R = [DR u; v];
