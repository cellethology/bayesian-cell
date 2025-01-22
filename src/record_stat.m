function [output] = record_stat(param,results,it)
%RECORD_PARAM Summary of this function goes here

avg_kfb = mean(results.kfb(1:it-1));
rcount = sum(results.f(2:it-1,:),2);
output = [avg_kfb,param.koff,mean(rcount),std(rcount)];

end

