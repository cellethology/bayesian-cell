function [schemerate,unifrate,statsummary] = racing_cells(fname, param, varargin)

p = inputParser;
isWithinRange = @(x) (x==0) || (x==1) || (x==2);
addRequired(p,'fname',@isstring);  % file containing fconc function
addRequired(p,'param',@isstruct);  % file containing scheme parameters

% optional arguments
addParameter(p,'save_data',2,isWithinRange); % 0 = no save, 1 = save without 
                                             % receptor and env profile
                                             % 2 = save all data
addParameter(p,'verbose',false,@islogical);
addParameter(p,'makeplot',true,@islogical);
addParameter(p,'task',"localization",@isstring); % localization, retention
addParameter(p,'envmodel',"tissue",@isstring); %tissue or grad
addParameter(p,'decoder_method',"optimal_noise",@isstring);
% options: optimal_noise,perfect,randomwalk
addParameter(p,'receptor',"feedback",@isstring);
% options: feedback, uniform
addParameter(p,'rununif',false,@islogical);
addParameter(p,'index',"",@isstring);
addParameter(p,'source',"edge",@isstring);
addParameter(p,'alpha',0,@isnumeric);

parse(p,fname,param,varargin{:});
fname = p.Results.fname;
param = p.Results.param;
save_data = p.Results.save_data;
isVerbose = p.Results.verbose;
makeplot = p.Results.makeplot;
task = p.Results.task;
envmodel = p.Results.envmodel;
decoder_method = p.Results.decoder_method;
receptor = p.Results.receptor;
rununif = p.Results.rununif;
index = p.Results.index;
source = p.Results.source;
alpha = p.Results.alpha;

% an unrecognised scheme has no branch in cell_move and would silently leave
% the receptor profile at zero, so fail loudly instead
validReceptors = ["feedback","uniform","bayes"];
if ~ismember(receptor, validReceptors)
    error("racing_cells:badReceptor", "receptor must be one of: %s", ...
          strjoin(validReceptors, ", "))
end

% establish cell shape
m = param.N;
phi = linspace(pi,-pi+(2*pi)/m,m)';
phi = circshift(flipud(phi),m/2+1);
phi(1) = 0; % correct for minor numerical inaccuracy
if ~isequal(task, "retention")
    s = param.mean_cell_radius;
    param.cellboundary = s*[cos(phi),sin(phi)];
    param.dx = ellipsePerimeter(s,s)/m;
else % use elongated growth cone for retention test
    a=2;b=5;
    param.cellboundary = [a*cos(phi),b*sin(phi)];
    param.dx = ellipsePerimeter(a,b)/m;    
end

% building fconc
load(fname,'cbound','csol','xmin','xmax','ymin','ymax');
ctot = csol + cbound;
if isequal(source,"point")
    pos = combvec(linspace(xmin,xmax,size(ctot,1)),...
                     linspace(ymin,ymax,size(ctot,2)))';
    param.bounds = [[xmax,ymax];[xmin,ymin]];
else
    pos = combvec(linspace(1,xmax-xmin,size(ctot,1)),...
                linspace(1,ymax-ymin,size(ctot,2)))';
    param.bounds = [[xmax-xmin-5,ymax-ymin-5];[5,5]];
end
fenv = scatteredInterpolant(pos,ctot(:),'natural','linear');
fconc = @(x,y) fenv([x,y]);

% setting cell starting position
nrun = 100;
if isequal(source,"point")
    theta = linspace(0,2*pi*(1-1/nrun),nrun);
    yc = 90*sin(theta);
    xc = 90*cos(theta);
    startpt = [xc;yc]';
else
    yc = linspace(param.mean_cell_radius*10,ymax-ymin-param.mean_cell_radius*10,nrun);
    if isequal(task, "localization")
        xc = 50;
    elseif isequal(task, "retention")
        xc = 5;
    end
    startpt = combvec(xc,yc)';
end

% plotting env
conversion_factor = conc2count(param.mean_cell_radius,param.N);
if isequal(source,"point")
    coord = xmin:xmax;
    envcoord = combvec(coord,coord)';
    env = reshape(fconc(envcoord(:,1),envcoord(:,2)),...
                          length(coord),length(coord))';
else
    xmax = xc+10;
    envcoord = combvec(1:xmax,1:round(ymax-ymin))';
    env = reshape(fconc(envcoord(:,1),envcoord(:,2)),xmax,round(ymax-ymin))';
end

if min(env,[],'all') < 0
    warning('negative ligand value')
end
ftissue = @(x,y) fconc(x,y)*conversion_factor; %count
if isequal(envmodel, "tissue")
    param.fcount = ftissue;
    if isequal(receptor,'bayes')
        param.cmat = 0;
    end
end
% the pure gradient model is the alpha = 1 limit of the blend below
if isequal(envmodel, "grad")
    alpha = 1;
end
if alpha > 0
    % the fit is to the y-averaged x-profile, which only describes an edge
    % source; a point source has no such profile
    if isequal(source,"point")
        error("envmodel 'grad' and alpha > 0 require source 'edge'")
    end
    f = fit((1:xmax)',mean(env)','exp1');
    fgrad = @(x,y) f(x)*conversion_factor; %count
    if alpha >= 1
        param.fcount = fgrad;
    else
        param.fcount = @(x,y) alpha*fgrad(x,y) + (1-alpha)*ftissue(x,y);
    end
%     if makeplot
%         plot(1:xmax,f(1:xmax),1:xmax,mean(env(:,1:xmax)));
%     end
    param.cmat = 0;
end

if makeplot
    xmax = xc+20;
    envcoord = combvec(1:xmax,1:round(ymax-ymin))';
    env = reshape(param.fcount(envcoord(:,1),envcoord(:,2)),xmax,...
                                                round(ymax-ymin))';
    figure(str2num(index))
    imagesc(env);
    title(['mean conc = ',num2str(mean(env,'all'))])
    pause(0.01)
end


% recording statistics
time_size = floor(param.T/param.dt);
move_rate = floor(30/param.dt); %move cell every 30 seconds
recstat = zeros(nrun,4);
posScheme = zeros(time_size/move_rate,2,nrun);
if rununif
    posUnif = zeros(time_size/move_rate,2,nrun);
end
if save_data == 2
    recF = zeros(time_size/move_rate,param.N,nrun);
    recC = zeros(time_size/move_rate,param.N,nrun);
end

%% simulating cell movement
% disp(param)
parfor ii = 1:nrun
    newparam = param;
    newparam.cellp = startpt(ii,:);
    % uniform receptor
    if rununif
        results_unif = cell_move(newparam,'mode',task,'verbose',isVerbose,...
            'decoder_method',decoder_method,'receptor', "uniform",...
            "source",source);
        posUnif(:,:,ii) = results_unif.cellp(1:move_rate:time_size,:);
    end
    % specified receptor
    results = cell_move(newparam,'mode',task,'verbose',isVerbose,...
        'decoder_method',decoder_method,'receptor',receptor,...
            "source",source);
    posScheme(:,:,ii) = results.cellp(1:move_rate:time_size,:);
    
    if isequal(receptor,'feedback')
        recstat(ii,:) = results.stat;
    end
    
    if save_data == 2
        recF(:,:,ii) = results.f(1:move_rate:time_size,:);
        recC(:,:,ii) = results.env(1:move_rate:time_size,:);
    end
%     disp(ii)
end

% recorded statistics such as h*a, r_memb, std r_memb
if isequal(receptor,'feedback')
    statsummary = mean(recstat);
    disp(strcat('<ha>,k_off,r_mem,std r_memb = ', num2str(statsummary)));
end

%% computing success and error rate from trajectories
total_time = param.T/60;
if isequal(task, 'localization')
    if rununif
        unif_time = squeeze(sum(posUnif(:,1,:)~=0))/2;
        unifrate = sum(unif_time<total_time)/nrun*100;
    else
        unifrate = nan;
    end
    scheme_time = squeeze(sum(posScheme(:,1,:)~=0))/2; %number of minutes elapsed
    schemerate = sum(scheme_time<total_time)/nrun*100;
    if rununif
        disp(strcat("uniform success rate = ",num2str(unifrate),"%"))
    end
    disp(strcat(receptor," success rate = ",num2str(schemerate),"%"))
    if makeplot
    %plotting
        if rununif
            histogram(unif_time,30,'Normalization','probability')
        end
        hold on
        histogram(scheme_time,30,'Normalization','probability')
        hold off
    end
elseif isequal(task, 'retention')
    if rununif
        unif_traj = squeeze(posUnif(:,1,:));
        unifrate = squeeze(sum(unif_traj-1>=3,[1,2]))/numel(unif_traj)*100;
    else
        unifrate = nan;
    end
    scheme_traj = squeeze(posScheme(:,1,:));
    schemerate = squeeze(sum(scheme_traj-1>=3,[1,2]))/numel(scheme_traj)*100;
    if rununif 
        disp(strcat('uniform error rate = ',num2str(unifrate),"%"))
    end
    disp(strcat(receptor,' error rate = ',num2str(schemerate),"%"))
    if makeplot
        %plotting
        if rununif 
            histogram(unif_traj,'Normalization','probability')
        end
        hold on;
        histogram(scheme_traj,'Normalization','probability')
        hold off;
    end
end
pause(0.001)

%% saving result
if isequal(envmodel, "grad")
    fname = strcat(fname,"_grad");
end
filename = strcat(fname,"_",task,"_",receptor);
if ~isequal(decoder_method,"optimal_noise")
    filename = strcat(filename,"_",decoder_method);
end
if save_data == 2
    save(strcat(filename,index),'-v7.3')
elseif save_data == 1
    save(strcat(filename,index))
end

end
