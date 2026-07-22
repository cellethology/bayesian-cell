function [results] = cell_move(param,varargin)

p = inputParser;
addRequired(p,'param',@isstruct);
addParameter(p,'verbose',false,@islogical);
addParameter(p,'mode',"static",@isstring);
% options: static, dynamic, localization (if static, decoder_method not
% needed)
addParameter(p,'decoder_method',"optimal_noise",@isstring);
% options: optimal_noise, perfect, randomwalk, circle
addParameter(p,'receptor',"uniform",@isstring);
% options: feedback, uniform, bayes
addParameter(p,'turncell',false,@islogical);
% whether to rotate cell 180 degrees half way through simulation
addParameter(p,'source',"edge",@isstring);
% options: edge, point

parse(p,param,varargin{:});
param = p.Results.param;
isVerbose = p.Results.verbose;
mode = p.Results.mode;
decoder_method = p.Results.decoder_method;
receptor = p.Results.receptor;
turncell = p.Results.turncell;
source = p.Results.source;

it = 1;
m = param.N;

cellp = param.cellp;
stepsz = param.stepsz;
fcount = param.fcount;
time_size = floor(param.T/param.dt);
move_rate = floor(30/param.dt); %move cell every 30 seconds
s = param.mean_cell_radius;

% Preallocate space
results.f = zeros(time_size,m);
results.cellp = zeros(time_size,2);
results.env = zeros(time_size,m);
results.a = zeros(time_size,m);

% Initial environment
results.cellp(it,:) = cellp;
cellboundary = param.cellboundary;
coord = cellp + cellboundary;
env = arrayfun(fcount,coord(:,1),coord(:,2))';
results.env(it,:) = env;
results.f(it,:)  = 0.9*param.rtot/m*ones(1,m);

if isequal(receptor, 'feedback')
    [param.L, param.R] = CNMatrix(param);
    results.FC = zeros(time_size,1);
    results.stat = zeros(1,4); 
    results.kfb = zeros(time_size,1);
    results.FC(it,:) = param.rtot - sum(results.f(it,:));
    %initialize param.hprop
    rvec = results.f(it,:);
    param.hprop = param.h/(mean(receptor_output(env,rvec,param)));
    results.kfb(it) = param.h;
    
elseif isequal(receptor, 'bayes')
    a = param.eps;
    if a>1
        error('motion model parameter must be less than or equal to 1')
    end
    if mod(m,2)==0
        x = linspace(-2,2,m+1);
    else
        x = linspace(-2,2,m);
    end
    fun = @(sigma) normpdf(0,0,abs(sigma))...
                /sum(normpdf(x,0,abs(sigma)))-(1-2*a);
    stdval = fzero(fun,0.01); %find the std that gives the desired eps
%     disp(["standard deviation = ",num2str(stdval)])
    if length(x) ~= m
        x = x(1:end-1);
    end
    px = circshift(normpdf(x,0,stdval),-ceil(m/2))/sum(normpdf(x,0,stdval));
    cmat = zeros(m,m);
    for ii = 1:m
        cmat(ii,:) = circshift(px,ii-1);
    end
%     cmat = diag((1-2*a)*ones(1,m))...
%                 + diag(a*ones(1,m-1),1)...
%                     + diag(a*ones(1,m-1),-1);
%     cmat(1,end) = a;
%     cmat(end,1) = a;
    param.cmat = cmat;
end

results.a(it,:) = receptor_output(env,results.f(it,:),param);

for it = 2:time_size  %it == "time iterators"
    % update receptors
    if isequal(receptor,"feedback")
        results = update_receptor(param,env,results,it);
    elseif isequal(receptor,"uniform")
        results.f(it,:) = results.f(1,:);
    elseif isequal(receptor,"bayes")
        rvec = results.f(it-1,:);
%         sample = receptor_output(env,rvec,param,"avgoutput",false);
        prior = rvec/sum(rvec);
%         sample = poissrnd(repmat(env,param.nsamp,1)); 
        sample = env;
        ppredict = bayeupdate(sample,prior,"cmat",param.cmat,...
                                           "beta",param.beta)';
        if isnan(sum(ppredict))
            error('Underflow or overflow, NaN error')
        end
        results.f(it,:) = ppredict./sum(ppredict)*param.rtot;
    end
    % move cell and update environment
    if ~isequal(mode,'static')
        if mod(it,move_rate) == 1
            rvec = results.f(it,:);
            ansum = receptor_output(env,rvec,param);
            cellp2 = nextpos(cellp, ansum, stepsz, decoder_method);
            
            %check if cell still inbound
            coord = cellp2 + cellboundary;
            envtemp = arrayfun(fcount,coord(:,1),coord(:,2))';
            
            if inbound(cellp2 + cellboundary,param.bounds)
                if ~all(envtemp>=0)
                    error("negative concentration sampled")
                end
                cellp = cellp2; %change cell position
                env = envtemp;
                if turncell && (it > round(time_size/2))
                    env = circshift(env,round(m/2));
                end
            end
            if isequal(mode,'localization')
                if isequal(source,'edge')
                    stopping = cellp(1) < 7+s;
                elseif isequal(source,'point')
                    stopping = norm(cellp) < 25+s;
                end
                if stopping % stop when cell is near source
                    results.cellp(it,:) = cellp;
                    results.env(it,:) = env;
                    if isequal(receptor,'feedback') %record summary statistics
                        results.stat = record_stat(param,results,it);
                    end
                    if isVerbose
                        disp(strcat('(',receptor,') Time taken: ', ...
                            num2str(floor(it*param.dt/60)),' mins'))
                    end
                    break
                end
            end
        end
        
    end
    results.cellp(it,:) = cellp;
    results.env(it,:) = env;
    
    if it == time_size %record summary statistics
        if isequal(receptor,'feedback')
            results.stat = record_stat(param,results,it);
        end
        if isequal(mode,'localization') && isVerbose
            disp(strcat("(",receptor,") ", "unfinished"))
        end
    end
end

end