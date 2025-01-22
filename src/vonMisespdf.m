function [pdf] = vonMisespdf(x,mu,kappa)
%VONMISESPDF Compute the probability density for a VonMises(mu,kappa)
pdf = exp(kappa*cos(x-mu))/(2*pi*besseli(0,kappa));
end

