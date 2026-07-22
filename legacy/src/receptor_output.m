function [a_samp] = receptor_output(env,rvec,param,varargin)
%receptor activity given environment, circuit parameters, and receptor
%arrangement

p = inputParser;
addRequired(p,'env',@isnumeric);
addRequired(p,'rvec',@isnumeric);
addRequired(p,'param',@isstruct);
addParameter(p,'avgoutput',true,@islogical);

parse(p,env,rvec,param,varargin{:});
env = p.Results.env;
rvec = p.Results.rvec;
param = p.Results.param;
avgoutput = p.Results.avgoutput;

if param.noisy
    c_samp = poissrnd(repmat(env,param.nsamp,1));
    rate = rvec.*hillfun(c_samp,param)/param.rtot;
    if avgoutput
        % A sum of independent Poissons is Poisson of the summed rate, so
        % drawing nsamp rows and averaging is distributionally identical to a
        % single row drawn at the summed rate -- but ~3x cheaper for this layer
        % (verified by KS test). The first layer cannot be collapsed this way
        % because hillfun is nonlinear. Note this consumes the RNG stream
        % differently, so realisations differ from the pre-change code even for
        % the same seed; the distribution does not.
        a_samp = poissrnd(sum(rate,1))/param.nsamp;
    else
        a_samp = poissrnd(rate);
    end
else
    a_samp = rvec.*hillfun(env,param)/param.rtot;
end

end

