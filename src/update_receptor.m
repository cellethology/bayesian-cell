function [results] = update_receptor(param, env, results, it)

if it == floor(10/param.dt)
    rvec = results.f(it-1,:);
    avgA = mean(receptor_output(env,rvec,param));
    param.hprop = param.h/avgA;
end

%1) update receptor activity profile
rvec = results.f(it-1,:);
results.a(it,:) = receptor_output(env,rvec,param);

%2) update Crank-Nicholson matricies
[L, R, meankfb] = UpdateCNMatrix(param,results,it);

%3) solve for next time step
v = [results.f(it-1,:)'; results.FC(it-1,:)];
C = L\R;
w = C*v;
results.f(it,:)  = w(1:end-1);
results.FC(it,:) = w(end);
results.kfb(it) = meankfb;

end

% -------------------------------------------------------------------------
function [L, R, meankfb] = UpdateCNMatrix(param,results,it)
% old L and R
L = param.L;
R = param.R;

% t-1 and t receptor activity masks are used for R and L (resp) matricies
mkActL = results.a(it,:);
mkActR = results.a(it-1,:);

% setting parameters
% kfb = param.hprop*an;
% meankfb = mean(kfb);
meankfb = mean(param.hprop.*mkActL);

% computing L and R
transportL   = - (param.hprop)/2*mkActL';
transportR   = + (param.hprop)/2*mkActR';
endocytosisL = (param.koff/2).*ones(1,param.N);
endocytosisR = -(param.koff./2).*ones(1,param.N);

L = L + diag([endocytosisL 0]);
L(1:end-1,end) = transportL;

R = R + diag([endocytosisR 0]);
R(1:end-1,end) = transportR;
end
