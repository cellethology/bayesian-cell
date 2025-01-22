function [output] = hillfun(input,params)
%computing average output assuming r = rtot
% output = (input./(input + params.kd) + ...
%     params.receptornoise.*params.kd./(input + params.kd))*params.rtot;

output = (input./(input + params.kd))*params.rtot;

end
