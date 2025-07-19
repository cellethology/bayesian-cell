function [initvalues] = make_init(YLEGI, RInit, baseMeanslope, baseMeanbasal, param)
% base withough noise
R = YLEGI(:,param.Np*2+1:end);
% code below simulate non-adaptive response
% E = YLEGI(:,1:Np);
% R = 1.25 + E;
base = (R - RInit*ones(size(R,1),size(R,2)))*baseMeanslope + baseMeanbasal;

% initial conditions: decided by initial R
polynomial0 = [param.alpha*(1+param.beta*param.gamma) -param.alpha*(param.a+1)-base(1,1) ...
    zeros(1,2*param.n-2) param.alpha*(1+param.beta*param.gamma)*param.a ...
    -base(1,1)*param.a];
rts = roots(polynomial0);
nrrts = 0;  % number of real roots
irrt = 0;
rrts = zeros(3,1);  % at most 3 real roots
for i = 1:2*param.n+1
    if imag(rts(i))==0
        nrrts = nrrts+1;
        irrt = i;
        rrts(nrrts) = real(rts(i));
    end
end
if nrrts == 1
    activator0 = rts(irrt);   
else    % nrrts == 3
    % select the lower steady state
    rrtssort = sort(rrts);
    activator0 = rrtssort(1);
end
activator0 = activator0*ones(1,param.Np);
inhibitor0 = param.gamma*activator0;

% Additional two components
componentZ0 = zeros(1,param.Np);
componentW0 = zeros(1,param.Np);

initvalues = [inhibitor0;activator0;componentZ0;componentW0];