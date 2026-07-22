function [posterior] = bayeupdate(data, prior, varargin)
%BAYEUPDATE compute probability of data given source at angle theta
%theta is a vector of angles
p = inputParser;
addRequired(p,'data',@ismatrix);
addRequired(p,'prior',@ismatrix);
addParameter(p,'cmat',1,@ismatrix);
addParameter(p,'beta',1,@isnumeric);

parse(p,data,prior,varargin{:});
data = p.Results.data;
prior = p.Results.prior;
cmat = p.Results.cmat;
beta = p.Results.beta;

% [nsample,nbin] = size(data);
% theta = linspace(-pi,pi-2*pi/nbin,nbin);
% logdataprob = zeros(1,nbin);

% if cmat~=0
%     [nloc,~] = size(cmat);
% end

% for ii = 1:nbin
%     if cmat~=0
%         temp = 0;
%         lambda = circshift(cmat,ii-1,2);
%         for jj = 1:nsample   
%             z = data(jj,:);
%             temp = temp + (-log(nloc)+...
%                 logsumexp(sum(log(poisspdf(repmat(z,nloc,1),lambda)),2)));
%         end
%         logdataprob(ii) = temp;
%     else
%         lambda = 0.5*exp(0.1478*cos(theta-theta(ii)));
%         logdataprob(ii) = sum(log(poisspdf(data,repmat(lambda,nsample,1))),'all');
%     end
% end

%motion model
priormove = cmat*prior';

%observation model
if size(data,1)>1
    logdataprob = sum(log(1+beta*data),1);
%     logdataprob = sum(800*data,1);
else
    logdataprob = log(1+beta*data);
%     logdataprob = 800*data;
end

if any(data<0)
    disp(data(data<0))
end
logposterior = logdataprob + log(priormove');

meanp = max(logposterior);
posterior = exp(logposterior-meanp);
posterior = posterior./sum(posterior);


end

