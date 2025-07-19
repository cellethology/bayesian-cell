function [timetaken,exitflag] = run_single(startpt,encoder,decoder,varargin)
p = inputParser;
addRequired(p,'startpt',@isvector);  
addRequired(p,'encoder',@isstring);
addRequired(p,'decoder',@isstring);

% optional arguments
addParameter(p,'makeplot',false,@islogical);
addParameter(p,'resultfold',"result",@isstring);
addParameter(p,'maxTime',5*60,@isnumeric);
addParameter(p,'savedata',true,@islogical);
addParameter(p,'gridmargin',5,@isnumeric);
addParameter(p,'cellrad',5,@isnumeric);
addParameter(p,'env',"tissue",@isstring);
addParameter(p,'update_grid',true,@islogical);
addParameter(p,'encoderparam',struct(),@isstruct);
addParameter(p,'source',"edge",@isstring);
addParameter(p,'fileindex',0,@isnumeric)

parse(p,startpt,encoder,decoder,varargin{:});
startpt = p.Results.startpt;
encoder = p.Results.encoder;
decoder = p.Results.decoder;

resultfold = p.Results.resultfold;
makeplot = p.Results.makeplot;
maxTime = p.Results.maxTime;
savedata = p.Results.savedata;
gridmargin  = p.Results.gridmargin;
Rcell  = p.Results.cellrad;
envmodel = p.Results.env;
update_grid = p.Results.update_grid;
encoderparam = p.Results.encoderparam;
source = p.Results.source;
fileindex = p.Results.fileindex;

timestep = 0.01;
Np = 314;

if isempty(fieldnames(encoderparam))
    encoderparam.cellradius = Rcell;
    
    if isequal(encoder,"receptor") || isequal(encoder,"uniform")
        encoderparam.rtot = 200;
        encoderparam.kd = 1;
        encoderparam.receptornoise = 0.1;
        encoderparam.noisy = true;
        encoderparam.N = Np;
        encoderparam.nsamp = 30*timestep;
    end
    
    if isequal(encoder,"receptor") 
        % load system parameters
        encoderparam.koff = 0.0678;
        encoderparam.h = 0.002;
        encoderparam.dt = timestep;
        encoderparam.dx = 2*pi*Rcell/Np;
        encoderparam.d = 0.01 * (100/Np)^2 *(Rcell/10)^2; %adjusted to match value from my paper
        encoderparam.T = maxTime;
        [encoderparam.L, encoderparam.R] = CNMatrix(encoderparam);
        
    elseif isequal(encoder,"LEGI_BEN_POL") 
        % https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003122
        load('LEGISysParam.mat');
        % parameters to decide the shape of x-nullcline
        encoderparam.xm = 0.1;
        encoderparam.n = 1;
        % tranfer into SDE system
        encoderparam.a = encoderparam.xm^(2*encoderparam.n-1)*(encoderparam.xm-1)...
                              /(encoderparam.xm^(2*encoderparam.n-1)-1); % 0.1
        encoderparam.alpha = palpha; % 2.8703
        encoderparam.beta = pbeta; % 3.7107
        encoderparam.gamma = pgamma; % 0.2143
        encoderparam.epsilon = pepsilon; % 0.0384
        encoderparam.Dactivator = 1.15 * Dactivator; % 2.1434 (X) this is only parameter different with Yuan's model (1.8639)                                     
        encoderparam.Dinhibitor = Dinhibitor; % 4.7729 (Y)
        encoderparam.DZ = 4;
        encoderparam.LEGIspdcoef = LEGIspdcoef;
        encoderparam.LEGIDI = LEGIinhibitorcoef;
        encoderparam.LEGIfrcoef = LEGIfrcoef;
        encoderparam.baseMeanslope = baseMeanslope;
        encoderparam.baseMeanbasal = baseMeanbasal;
    end
end
    
% building fconc
fname = "tissue_300by900.mat";
load(fname,'cbound','csol','xmin','xmax','ymin','ymax');
ctot = csol(5:end,:) + cbound(5:end,:);
pos = combvec(linspace(1,xmax-xmin,size(ctot,1)),...
                linspace(1,ymax-ymin,size(ctot,2)))';
fenv = scatteredInterpolant(pos,ctot(:),'natural','linear');
fconc = @(x,y) fenv([x,y]);

if isequal(envmodel, "expgrad")
    % make smooth gradient
    envcoord = combvec(1:60,101:700)';
    env = reshape(fconc(envcoord(:,1),envcoord(:,2)),60,600)';
    f = fit((1:60)',mean(env)','exp1');
    fconc = @(x,y) f(x); 
end

% geometry of the cell
if gridmargin ~= Rcell
    warning("grid margin does not equal cell radius")
end
phi = linspace(-pi,pi-(2*pi)/Np,Np)';
cellboundary = Rcell*[cos(phi),sin(phi)];
coord = startpt + cellboundary;

% specify timeRange for the solver
moverate = timestep*30; %move every [moverate] seconds
timeRange = 0:timestep:moverate;
stepRange = moverate:moverate:maxTime;
tlength = length(timeRange)-1;
totTimeStep = length(stepRange)*length(timeRange)+1;

if isequal(decoder,"LSM")
    % The cell's mechanics properties
    K = 0.098;%(pi*Rp^2); %4.76 s*nN  all divided by um^2
    D = 0.064;%(pi*Rp^2);  %2.94 s*nN/um all divided by um^2
    B = 6.09;%(pi*Rp^2);  %298.8 s*nN/um all divided by um^2
    volumeConst = 1; % originally 1
    surfTension = 1; % nN/um
    gs = 1; % Grid numbers per um used to be 19!! 
            % when there are 360 points need to be at least 17

    % Set up level sets
    % Sets up grid dimensions automatically based on cell size.
    xleft_grid      = -(round(Rcell) + gridmargin);
    xright_grid     = round(Rcell) + gridmargin;
    ybottom_grid    = -(round(Rcell) + gridmargin);
    ytop_grid       = round(Rcell) + gridmargin;
    levelSetConfiguration = setupLevelSets('low', xleft_grid, xright_grid,... 
                             ybottom_grid, ytop_grid, gs, volumeConst,...
                             surfTension, K, D, B, Rcell);   
    schemeData = levelSetConfiguration.schemeData;
    g = schemeData.grid;
    gridcenter = (g.min+g.max)/2;
    % Create Initial Membrane Potential Function
    % rows of membranePotentialFunction represent x and col represent y-axis
    membranePotentialFunction = shapeSphere(g, [0 0],Rcell);

    % Initialize schemeData values
    schemeData = Initialization(schemeData, membranePotentialFunction,levelSetConfiguration);
    
    % Initialize springs and x and y potential functions
    spring = zeros(g.shape); 
    tNow = 0; % unit = 100 seconds
    
elseif isequal(decoder,"simple")
    cellp = startpt;
    stepsz = 2*moverate/60; %micron per step
    movdir = zeros(length(stepRange),1);
    schemeData = struct('stepsz',stepsz,'startpt',startpt);
end


if isequal(encoder,"uniform")
    runif = ones(1,encoderparam.N)*encoderparam.rtot/encoderparam.N;
    
elseif isequal(encoder,"receptor")     
    % Preallocate space
    results = struct();
    results.FC = zeros(totTimeStep,1);
    results.f = zeros(totTimeStep,Np);
    results.a = zeros(totTimeStep,Np);
    results.env = {};
    results.kfb = zeros(totTimeStep,1);
    
    % Initial receptor conditions
    it = 1;
    results.f(it,:)  = 0.9*encoderparam.rtot/Np*ones(1,Np);
    rvec = results.f(it,:);
    results.FC(it,:) = encoderparam.rtot - sum(results.f(it,:));
    env = arrayfun(fconc,coord(:,1),coord(:,2))';
    results.a(it,:) = receptor_output(env,rvec,encoderparam);
    encoderparam.hprop = encoderparam.h/(mean(results.a(it,:)));
    results.kfb(it) = encoderparam.h;
    
elseif isequal(encoder,"LEGI_BEN_POL")
    % system parameters
    encoderparam.Np = Np;
    LEGIinput.Np = Np;
    LEGIparam.Np = Np;
    LEGIparam.kfe = 10*encoderparam.LEGIspdcoef; % 0.5
    LEGIparam.kre = 10*encoderparam.LEGIspdcoef; % 0.5
    LEGIparam.kfi = encoderparam.LEGIDI*0.025*encoderparam.LEGIspdcoef; % 0.1
    LEGIparam.kri = encoderparam.LEGIDI*0.025*encoderparam.LEGIspdcoef; % 0.1
    LEGIparam.kfr = encoderparam.LEGIfrcoef*2*encoderparam.LEGIspdcoef; % 0.06
    LEGIparam.krr = 2*encoderparam.LEGIspdcoef; % 0.1
    LEGIparam.Rtotal = 2;
    if encoderparam.LEGIDI == 0
        % set an arbitrary initial response
        RInit = LEGIparam.Rtotal*(LEGIparam.kre/LEGIparam.kfe)...
            /(LEGIparam.kfr/LEGIparam.krr+LEGIparam.kre/LEGIparam.kfe);
    else
        RInit = LEGIparam.Rtotal*(LEGIparam.kfi*LEGIparam.kre/(LEGIparam.kri*LEGIparam.kfe))...
            /(LEGIparam.kfr/LEGIparam.krr+LEGIparam.kfi*LEGIparam.kre/(LEGIparam.kri*LEGIparam.kfe));
    end
    %initial conditions
    R0 = RInit*ones(1,Np);
    I0 = zeros(1,Np);
    E0 = zeros(1,Np);
    YLEGI_0 = [E0'; I0'; R0']';
    YLEGI = zeros(length(stepRange)*tlength+1,3*LEGIparam.Np);
    TLEGI = zeros(1,length(stepRange)*tlength+1);
    Y = zeros(length(stepRange)*tlength+1,4*LEGIparam.Np);
end


%% running simulation
try
for ii = 1:length(stepRange)
   externalSignal = arrayfun(fconc,coord(:,1),coord(:,2))';
   if isequal(decoder,"LSM")
       membraneToCenterX = cellboundary(:,1)' - schemeData.centerX(end);
       membraneToCenterY = cellboundary(:,2)' - schemeData.centerY(end);
       angleToCenter = atan2(membraneToCenterY,membraneToCenterX);
       if angleToCenter(end) == angleToCenter(1)
           angleToCenter = angleToCenter(1:end-1);
           externalSignal = externalSignal(1:end-1);
       end
       signalAngle = -pi:(2*pi)/(Np-1):pi;
       signalAngle(end) = signalAngle(end) + 0.002;
       externalSignal = interp1([angleToCenter-2*pi,angleToCenter,angleToCenter+2*pi],...
           [externalSignal,externalSignal,externalSignal],...
           signalAngle,'nearest');
   end
   
   if isequal(encoder,"LEGI_BEN_POL")
       timeRange = (ii-1)*moverate:timestep:ii*moverate;
       LEGIinput.base = externalSignal';
       [TLEGIpart,YLEGIpart] = ode45(@LEGI_EQS,timeRange,YLEGI_0,[],LEGIinput,LEGIparam);
       E0 = YLEGIpart(end,1:LEGIparam.Np);
       I0 = YLEGIpart(end,LEGIparam.Np+1:2*LEGIparam.Np);
       R0 = YLEGIpart(end,2*LEGIparam.Np+1:end);
       YLEGI_0 = [E0'; I0'; R0']';
       if ii == 1
           timeindex = 1:(tlength+1);
           YLEGI(timeindex,:) = YLEGIpart;
           TLEGI(timeindex) = TLEGIpart;
           %%initial EN and POL condition
           initvalues = make_init(YLEGIpart,RInit,encoderparam.baseMeanslope,...
                                    encoderparam.baseMeanbasal,encoderparam);
       else
           timeindex = 2+(ii-1)*tlength:ii*tlength+1;
           YLEGI(timeindex,:) = YLEGIpart(2:end,:);
           TLEGI(timeindex) = TLEGIpart(2:end,:);
           %%initial EN and POL condition
           initvalues = reshape(Ypart(end,:),4,Np);
       end
       Ypart = BEN_POL(initvalues,TLEGIpart,YLEGIpart,RInit,timeRange,encoderparam);
       if ii == 1
           Y(timeindex,:) = Ypart;
       else
           Y(timeindex,:) = Ypart(2:end,:);
       end
       outputsignal = Y(timeindex,1:4:end);
   elseif isequal(encoder,"receptor")
       %move receptors
       for jj = 1:tlength+1
           it = (ii-1)*(tlength+1)+jj+1;
           results = update_receptor(encoderparam,externalSignal,results,it);
       end
       timeindex = (ii-1)*(tlength+1)+(1:tlength+1)+1;
       outputsignal = results.a(timeindex,:);
   elseif isequal(encoder,"uniform")
       outputsignal = zeros(tlength+1,encoderparam.N);
       for kk = 1:tlength+1
           outputsignal(kk,:) = receptor_output(externalSignal,runif,encoderparam);
       end
   end
    
   if isequal(decoder,"simple")
       dir = grad_decode(outputsignal(end,:), "optimal_noise") + pi;
       movdir(ii) = dir;
       cellp = cellp + stepsz*[cos(dir), sin(dir)];
       coord = cellp + cellboundary;
       
   elseif isequal(decoder,"LSM")
       [tNow,schemeData,membranePotentialFunction,~,levelSetConfiguration] =...
           LSM(outputsignal,tNow,schemeData,membranePotentialFunction,...
           spring,timestep,levelSetConfiguration);
       
       if update_grid
           % update grid over which potential function is computed
           deltax = schemeData.centerX(end) - gridcenter(1);
           deltay = schemeData.centerY(end) - gridcenter(2);
           changex = abs(deltax) > schemeData.grid.dx(1);
           changey = abs(deltay) > schemeData.grid.dx(2);
           if changex
               shiftx = round(deltax/schemeData.grid.dx(1))*schemeData.grid.dx(1);
               xleft_grid    = xleft_grid + shiftx;
               xright_grid   = xright_grid + shiftx;
           end
           if changey
               shifty = round(deltay/schemeData.grid.dx(2))*schemeData.grid.dx(2);
               ytop_grid    = ytop_grid + shifty;
               ybottom_grid   = ybottom_grid + shifty;
           end
           if changex || changey
               g_old = schemeData.grid;
               g = make_grid(schemeData,gs,xleft_grid,xright_grid,ybottom_grid,ytop_grid);
               schemeData.grid = g;
               levelSetConfiguration.schemeData.grid = g;
               memSpring = schemeData.memSpring{end};
               membranePoints = schemeData.membranePoints{end};
               % update spring
               spring = griddata(membranePoints(1,:), membranePoints(2,:), ...
                                    memSpring, g.xs{1}, g.xs{2}, 'nearest');
               membranePotentialFunction = interp2(g_old.xs{1}',g_old.xs{2}',...
                                    membranePotentialFunction',g.xs{1}',g.xs{2}')';
               membranePotentialFunction = inpaint_nans(membranePotentialFunction);
               gridcenter = (g.min+g.max)/2;
           end
       end
       cellboundary = schemeData.membranePoints{end}';
       coord = startpt + cellboundary;
       
       %terminate run if cell passed environment boundary
       if ~inbound(coord,source)
           fprintf('\nTerminated! (%g,%g)',startpt(1),startpt(2));
           exitflag = 2;
           break
       end
   end
       
   leftTip = min(coord(:,1));
   if mod(ii,round(6000/moverate)) == 0
       fprintf('\nTime elapsed: %g minutes, left edge = %g',...
                 round(ii*moverate/60,2),...
                 round(leftTip,1));
   end

   % check if cell finished moving
   if leftTip < 3
       fprintf('\nSuccess! Time elapsed: %g minutes',round(ii*moverate/60,2))
       exitflag = 1;
       break
   elseif ii == length(stepRange)
       fprintf('\nFailed! (%g,%g)',startpt(1),startpt(2));
       exitflag = 0;
       break
   end
end
catch ME
    exitflag = 3;
    disp(getReport(ME,'extended','hyperlinks','on'))
end

% record time taken in minutes
if exitflag ~= 3
    timetaken = ii*moverate/60;
else
    timetaken = nan;
end

%% postprocessing

%save result
if savedata 
    % rename if the result folder already exists
    if ~exist(resultfold,'dir')
        mkdir(resultfold);
    end
    
    savename = strcat(resultfold,"/startpt_",num2str(round(startpt(2))));
    if fileindex > 0
        savename = strcat(savename,"_",num2str(fileindex));
    end
    
    % only save every 100 frames (30 seconds) to save space
    downsamp = 300;
    schemeData.downsamp = downsamp;
    schemeData.membranePoints = schemeData.membranePoints(1:downsamp:end);
    schemeData.memSpring = schemeData.memSpring(1:downsamp:end);
    schemeData.surfArea = schemeData.surfArea(1:downsamp:end);
    schemeData.perimeter = schemeData.perimeter(1:downsamp:end);
    schemeData.centerX = schemeData.centerX(1:downsamp:end);
    schemeData.centerY = schemeData.centerY(1:downsamp:end);
    schemeData.time = schemeData.time(1:downsamp:end);
    schemeData.n = length(schemeData.time)+1;
    
    if isequal(encoder,"receptor")
        save(strcat(savename,'.mat'),'schemeData','results','maxTime',...
            'timetaken','encoderparam','fconc','timestep',...
            'moverate','timeRange','stepRange',...
            'totTimeStep','startpt','exitflag','envmodel');
    elseif isequal(encoder,"uniform")
        save(strcat(savename,'.mat'),'schemeData','maxTime',...
            'timetaken','encoderparam','fconc','timestep',...
            'moverate','timeRange','stepRange',...
            'totTimeStep','startpt','exitflag','envmodel');
    elseif isequal(encoder,"LEGI_BEN_POL")
%         ntimeperstep = round(moverate/timestep);
%         index = ntimeperstep:ntimeperstep:(maxTime+1);
%         TLEGI_subset = TLEGI(index);
%         YLEGI_subset = YLEGI(index,:);
%         Y_subset = Y(index,:);
        save(strcat(savename,'.mat'),'schemeData','maxTime',...
            'timetaken','encoderparam','fconc','timestep',...
            'moverate','timeRange','stepRange',...
            'totTimeStep','startpt','exitflag','envmodel');
    end
end

if makeplot 
    if isequal(encoder,"receptor")
        % plot LEGI kymograph
        tiledlayout(2,1);
        nexttile
        imagesc(circshift(results.f(1:it,:),Np/2,2)'); %receptor
        title("Receptor")
        nexttile
        imagesc(circshift(results.a(1:it,:),Np/2,2)'); %activity
        title("activity")
        if savedata
            saveas(gcf,strcat(savename,'_kymo.tif'), 'tiffn')
        end
      
    elseif isequal(encoder,"LEGI_BEN_POL")
        % plot LEGI kymograph
        tiledlayout(3,1);
        nexttile
        imagesc(circshift(YLEGI(:,1:LEGIparam.Np),Np/2,2)'); %E
        title("E")
        nexttile
        imagesc(circshift(YLEGI(:,LEGIparam.Np+1:2*LEGIparam.Np),Np/2,2)'); %I
        title("I")
        nexttile
        imagesc(circshift(YLEGI(:,2*LEGIparam.Np+1:end),Np/2,2)'); %RR
        title("R")
        if savedata
            saveas(gcf,strcat(savename,'_kymo.tif'), 'tiffn')
            % plot BEN-POL kymograph
            plot_BEN_POL(Y,timestep,maxTime,Np,savename)
        end
        
    end
end

close(gcf)

        

%% System equation for LEGI

function dydt = LEGI_EQS(t,y,input,param)

E = y(1:input.Np);
I = y(input.Np+1:2*input.Np);
R = y(2*input.Np+1:end);

% get input
IN = getLEGIInput_grdt(t,input);

dE = param.kre * IN - param.kfe * E;
dI = param.kri * IN - param.kfi * I;
dI = mean(dI)*ones(size(dI,1),size(dI,2));
dR = -param.kfr * R .* I + param.krr * (param.Rtotal*ones(param.Np,1)-R).* E;

dydt = [dE; dI; dR];

%% LEGI input

function IN = getLEGIInput_grdt(t,input)

% if t < input.tON
%     IN = zeros(input.Np,1);
% else
%     IN = input.base';
% end
IN = input.base;

%% Output

function I = PlotResult1D(I1data,I2data,I3data,I1min,I1max,I2min,I2max,I3min,I3max,...
    I0,idxbd,Intensitymin,Intensitymax)

[h,w] = size(I0);
I = uint8(zeros(h,3*w));
I1 = uint8(zeros(h,w));
I2 = uint8(zeros(h,w));
I3 = uint8(zeros(h,w));
I1(idxbd) = Intensitymin + (Intensitymax-Intensitymin)*(I1data-I1min)/(I1max-I1min);
I2(idxbd) = Intensitymin + (Intensitymax-Intensitymin)*(I2data-I2min)/(I2max-I2min);
I3(idxbd) = Intensitymin + (Intensitymax-Intensitymin)*(I3data-I3min)/(I3max-I3min);
I(:,1:w) = I1;
I(:,w+1:2*w) = I2;
I(:,2*w+1:end) = I3;


