function [output] = inbound(cellcoord,bounds)

% check if cell is inside domain
output = all([max(cellcoord)<bounds(1,:),min(cellcoord)>bounds(2,:)]);


end

