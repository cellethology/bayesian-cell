function [results] = update_receptor(param, env, results, it)

%0) signal-coupled diffusivity (optional): ligand occupancy suppresses lateral
%   mobility, so the prior-blur term shrinks where the signal is strong. param
%   is passed by value, so d is recomputed from the baseline every timestep
%   rather than ratcheting down.
if isfield(param,'dcouple') && param.dcouple
    z = env./(env + param.kd);                   % Hill occupancy, in [0,1)
    d = param.d./(1 + (z/param.z0).^param.dn);   % per-bin, decreasing in z
    if isfield(param,'dmin')
        % floor the diffusivity so the belief retains enough mobility to track
        % a bearing that rotates as the cell advances. Analogous to the clip on
        % sigma_Q in the Python coupled EKF (filters/base_filter.py).
        d = max(d, param.dmin);
    end
    param.d = d;
    [param.L, param.R] = CNMatrix(param);
end

%1) update receptor activity profile
rvec = results.f(it-1,:);
results.a(it,:) = receptor_output(env,rvec,param);

%2) update Crank-Nicholson matricies
[L, R, meankfb] = UpdateCNMatrix(param,results,it);

%3) solve for next time step
% L\(R*v) rather than (L\R)*v: mathematically identical (verified bit-identical
% on a full run), but forms the matrix-vector product first instead of
% back-substituting N+1 right-hand sides and then discarding all but one
% combination. ~3x cheaper in isolation, though the solve is not the loop's
% bottleneck: measured 1.25x end-to-end deterministic, ~1.0x once Poisson
% sampling is on.
v = [results.f(it-1,:)'; results.FC(it-1,:)];
w = L\(R*v);
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
