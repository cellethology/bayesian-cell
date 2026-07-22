function [cellp] = nextpos(cellp, activity, stepsz, decoder_method)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

m = length(activity);

if mod(m,2) ~= 0 && isequal(decoder_method, 'perfect')
    error('number of membrane bin should be even')
end

phi = linspace(pi,-pi+(2*pi)/m,m)';
phi = circshift(flipud(phi),m/2+1);
phi(1) = 0; % correct for minor numerical inaccuracy

if isequal(decoder_method, "optimal_noise")
    z1 = sum(cos(phi).* activity');
    z2 = sum(sin(phi).* activity');
    angle = atan2(z2,z1) + randn(1)*0.1;
elseif isequal(decoder_method, "perfect")
    [~,I] = max(activity - circshift(activity, m/2));
    angle = phi(I);
elseif isequal(decoder_method, "randomwalk")
    angle = phi(randi([1,m]));
elseif isequal(decoder_method, "circle") 
    %moving in a straight line in y-direction
%     angle = phi(m/4+1); % pi/2
    %move in a circle
    angle = atan2(-cellp(1),cellp(2));
end

cellp = cellp + stepsz*[cos(angle), sin(angle)];

end

