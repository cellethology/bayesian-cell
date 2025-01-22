function [L R] = CNMatrix(param)
% Notes:
% 1) we will eventually solve (Left and Right sides):
%    Lw(t+1) = Rw(t)
%    where the vector w = [f1;...;fN;F_cyto]
% 2) build Crank-Nicholson matrix with L & R = [D u; v 1], where
%    D is param.N x param.N
%    u is 1 x param.N
%    v is param.N x 1

%1) useful constants
N = param.N;
alpha = param.d/(2*param.dx^2);
betaL  = 1/param.dt + 2*alpha; 
betaR  = 1/param.dt - 2*alpha;

%2) diagonal for Left (DL) and Right (DR) sides
DL = -diag(alpha*ones(N-1,1),-1) + diag(betaL*ones(N,1)) - diag(alpha*ones(N-1,1),1);
% assume circular space
DL(1,end) = -alpha;
DL(end,1) = -alpha;

DR = diag(alpha*ones(N-1,1),-1) + diag(betaR*ones(N,1)) + diag(alpha*ones(N-1,1),1);
% assume circular space
DR(1,end) = alpha;
DR(end,1) = alpha;

%3) u,v
u = zeros(N,1); % u update in solver
v = ones(1,N+1);  % 

%4)
L = [DL u; v];
R = [DR u; v];
