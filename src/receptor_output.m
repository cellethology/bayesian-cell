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
    a_samp = poissrnd(rvec.*hillfun(c_samp,param)/param.rtot);
    if avgoutput
        a_samp = mean(a_samp,1);
    end
else
    a_samp = rvec.*hillfun(env,param)/param.rtot;
end

end

