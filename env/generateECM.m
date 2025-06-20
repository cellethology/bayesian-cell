function [ecmpts] = generateECM(N)
%generates an Mx2 array with the position of N fibers, where M is the
%number of fiber segments of length 0.5um
%   fiber length is normally distribution with mean 75 and stdev 5
sz = 75+5*randn(1,N); %N normally distributed fiber length
X1 = 2400*rand(1,N);  %N fiber position where each location has equal probability
Y1 = 1200*rand(1,N);
angle = 2*pi.*rand(1,N); % orientation of fiber uniformly distributed
X2 = X1 + sz.*cos(angle);
Y2 = Y1 + sz.*sin(angle);
plot([X1; X2], [Y1; Y2],'k')

ecmpts = zeros(2000000,2);
lenspt=1;
% divide up each fiber into equal-sized segments of length 0.5um
for ii = 1:N
    segment = 0:0.5:sz(ii);
    pts = [X1(ii)+segment.*cos(angle(ii));Y1(ii)+segment.*sin(angle(ii))]';
    ind2 = lenspt + size(pts,1)-1;
    ecmpts(lenspt:ind2,:) = pts;
    lenspt = lenspt + size(pts,1);
end

end

