function [timetaken] = LEGI_BEN_POL_LSM(startpt,varargin)
p = inputParser;
addRequired(p,'startpt',@isvector);  

% optional arguments
addParameter(p,'makeplot',true,@islogical);
addParameter(p,'fold0',"result",@isstring);
addParameter(p,'maxTime',60*2,@isnumeric);
addParameter(p,'savedata',true,@islogical);
addParameter(p,'useLSM',true,@islogical);
addParameter(p,'update_grid',true,@islogical);

parse(p,startpt,varargin{:});
update_grid = p.Results.update_grid;
startpt = p.Results.startpt;
fold0 = p.Results.fold0;
makeplot = p.Results.makeplot;
maxTime = p.Results.maxTime;
savedata = p.Results.savedata;
useLSM = p.Results.useLSM;

timestep = 0.1;
% frameRate = 10;
Np = 314;

% load system parameters
load('../../SysParam.mat');

%% %%%%%% simulate gradient response from LEGI model

% parameters to decide the shape of x-nullcline
xm = 0.1;
n = 1;

% tranfer into SDE system
param.Np = Np;
param.n = n;
param.a = xm^(2*n-1)*(xm-1)/(xm^(2*n-1)-1); % 0.1
param.alpha = palpha; % 2.8703
param.beta = pbeta; % 3.7107
param.gamma = pgamma; % 0.2143
param.epsilon = pepsilon; % 0.0384
param.Dactivator = 1.15 * Dactivator; % 2.1434 this is only parameter different with Yuan's model (1.8639)                                     
param.Dinhibitor = Dinhibitor; % 4.7729  

% system parameters
LEGIinput.Np = Np;
LEGIparam.Np = Np;

LEGIparam.kfe = 10*LEGIspdcoef; % 0.5
LEGIparam.kre = 10*LEGIspdcoef; % 0.5
LEGIparam.kfi = LEGIinhibitorcoef*0.025*LEGIspdcoef; % 0.1
LEGIparam.kri = LEGIinhibitorcoef*0.025*LEGIspdcoef; % 0.1
LEGIparam.kfr = LEGIfrcoef*2*LEGIspdcoef; % 0.06
LEGIparam.krr = 2*LEGIspdcoef; % 0.1

LEGIparam.Rtotal = 2;
if LEGIinhibitorcoef == 0
    % set an arbitrary initial response
    RInit = LEGIparam.Rtotal*(LEGIparam.kre/LEGIparam.kfe)...
        /(LEGIparam.kfr/LEGIparam.krr+LEGIparam.kre/LEGIparam.kfe);
else
    RInit = LEGIparam.Rtotal*(LEGIparam.kfi*LEGIparam.kre/(LEGIparam.kri*LEGIparam.kfe))...
        /(LEGIparam.kfr/LEGIparam.krr+LEGIparam.kfi*LEGIparam.kre/(LEGIparam.kri*LEGIparam.kfe));
end

% geometry of the cell 
Rcell = 7;  % cell radius
phi = linspace(-pi,pi-(2*pi)/Np,Np)';
cellboundary = Rcell*[cos(phi),sin(phi)];

% The cell's mechanics properties
K = 0.098;%(pi*Rp^2); %4.76 s*nN  all divided by um^2
D = 0.064;%(pi*Rp^2);  %2.94 s*nN/um all divided by um^2
B = 6.09;%(pi*Rp^2);  %298.8 s*nN/um all divided by um^2
volumeConst = 1; % originally 1
surfTension = 1; % nN/um
gs = 2; % Grid numbers per um used to be 19!! % when there are 360 points need to be at least 17

% Set up level sets

% Sets up grid dimensions automatically based on cell size.
extra = 5;
xleft_grid      = -(round(Rcell) + extra);
xright_grid     = round(Rcell) + extra;
ybottom_grid    = -(round(Rcell) + extra);
ytop_grid       = round(Rcell) + extra;
levelSetConfiguration = setupLevelSets('low', xleft_grid, xright_grid,... 
                             ybottom_grid, ytop_grid, gs, volumeConst,...
                                          surfTension, K, D, B, Rcell);                               
levelSetConfiguration.schemeData.gs = gs;
schemeData = levelSetConfiguration.schemeData;
g = schemeData.grid;

% Create Initial Membrane Potential Function
membranePotentialFunction = shapeSphere(g, [0 0],Rcell);

% Initialize schemeData values
schemeData = Initialization(schemeData, membranePotentialFunction,levelSetConfiguration);

% Initialize springs and x and y potential functions
spring = zeros(g.shape); 
tNow = 0; % unit = 100 seconds

%initial conditions
R0 = RInit*ones(1,Np);
I0 = zeros(1,Np);
E0 = zeros(1,Np);
YLEGI_0 = [E0'; I0'; R0']';

% specify timeRange for the solver
moverate = 1; %move every [moverate] seconds
stepRange = moverate:moverate:maxTime;
timeRange = 0:timestep:moverate;
tlength = length(timeRange)-1;

% building fconc
fname = "/Users/jerrywang/Documents/OneDrive - California Institute of Technology/SpatialComp/receptor-code/figure_2/data/tissue_env/default_param_env/tissue_300by900.mat";
load(fname,'cbound','csol','xmin','xmax','ymin','ymax');
ctot = csol(5:end,:) + cbound(5:end,:);
pos = combvec(linspace(1,xmax-xmin,size(ctot,1)),...
                linspace(1,ymax-ymin,size(ctot,2)))';
fenv = scatteredInterpolant(pos,ctot(:),'natural','linear');
% fconc = @(x,y) fenv([x,y])/globalmu * mean(LEGIinput.base);
% fconc = @(x,y) fenv([x,y]) * 3.6858;
fconc = @(x,y) fenv([x,y]);
% make smooth gradient
envcoord = combvec(1:60,201:700)';
env = reshape(fconc(envcoord(:,1),envcoord(:,2)),60,500)';
f = fit((1:60)',mean(env)','exp1');
fgrad = @(x,y) f(x); 

% cell position
cellp = zeros(length(stepRange)+1,2);
cellp(1,:) = startpt;

% solve
YLEGI = zeros(length(stepRange)*tlength+1,3*LEGIparam.Np);
TLEGI = zeros(1,length(stepRange)*tlength+1);
Y = zeros(length(stepRange)*tlength+1,4*LEGIparam.Np);

% coord = startpt + schemeData.membranePoints{end}';
% externalSignal = arrayfun(fconc,coord(:,1),coord(:,2));
% plot(externalSignal,'r');
% disp(size(externalSignal))
% pause(5)
for ii = 1:length(stepRange)
    timeRange = (ii-1)*moverate:timestep:ii*moverate;
%     coord = cellp(ii,:) + cellboundary;
    mempt = schemeData.membranePoints{end}';
    coord = startpt + mempt;
    externalSignal = arrayfun(fconc,coord(:,1),coord(:,2))';
    membraneToCenterX = mempt(:,1)' - schemeData.centerX(end);
    membraneToCenterY = mempt(:,2)' - schemeData.centerY(end);
    angleToCenter = atan2(membraneToCenterY,membraneToCenterX);
    if angleToCenter(end) == angleToCenter(1)
        angleToCenter = angleToCenter(1:end-1);
        externalSignal = externalSignal(1:end-1);
    end
    signalAngle = [-pi:(2*pi)/313:pi];
    signalAngle(end) = signalAngle(end) + 0.002;
    membraneSignal = interp1([angleToCenter-2*pi,angleToCenter,angleToCenter+2*pi],...
                             [externalSignal,externalSignal,externalSignal],...
                              signalAngle,'nearest');
    LEGIinput.base = membraneSignal';
    
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
        initvalues = make_init(YLEGIpart,RInit,baseMeanslope,baseMeanbasal,param);
    else
        timeindex = 2+(ii-1)*tlength:ii*tlength+1;
        YLEGI(timeindex,:) = YLEGIpart(2:end,:);
        TLEGI(timeindex) = TLEGIpart(2:end,:);
        %%initial EN and POL condition
        initvalues = reshape(Ypart(end,:),4,Np);
    end
    
    Ypart = BEN_POL(initvalues,TLEGIpart,YLEGIpart,RInit,timeRange,param);
    
    if ii == 1
        Y(timeindex,:) = Ypart;
    else
        Y(timeindex,:) = Ypart(2:end,:);
    end
 
    Ysignal = Y(timeindex,1:4:end);
    [tNow,schemeData,membranePotentialFunction,spring,levelSetConfiguration] =...
        LSM(Ysignal,tNow,schemeData,membranePotentialFunction,...
                        spring,timestep,levelSetConfiguration);
    % update grid
    g_old = schemeData.grid;
    g = make_grid(schemeData);
    schemeData.grid = g;
    levelSetConfiguration.schemeData.grid = g;
    
%     g.xs{2};
%     if mod(ii,2) == 0
%         fprintf('\nTime elapsed: %g minutes, at position: %g',...
%             ii*moverate/60,...
%             round(schemeData.centerX(end),1));
%     end

    if min(coord(:,1)) < startpt(1) - 40
        fprintf('Time elapsed: %g minutes',ii*moverate/60)
        break
    elseif ii == length(stepRange)
        fprintf('Run failed')
    end
end
timetaken = ii;

%% postprocessing
if ~exist(fold0,'dir')
    mkdir(fold0);
end

% % save result
if savedata
    savename = strcat(fold0,"/chemo_",num2str(round(startpt(2))));
    save(strcat(savename,'.mat'),'TLEGI','YLEGI','RInit','Y',...
                    'timestep','maxTime','Np','timetaken','schemeData');
end

if makeplot 
    % plot LEGI kymograph
    tiledlayout(3,1);
    nexttile
    imagesc(YLEGI(:,1:LEGIparam.Np)'); %E
    % saveas(gcf,strcat(savename,'.tif'), 'tiffn')
    nexttile
    imagesc(YLEGI(:,LEGIparam.Np+1:2*LEGIparam.Np)'); %I
    nexttile
    imagesc(YLEGI(:,2*LEGIparam.Np+1:end)'); %RR
    saveas(gcf,strcat(savename,'_kymo.tif'), 'tiffn')
    close(gcf)

    % plot BEN-POL kymograph
    plot_BEN_POL(Y,timestep,maxTime,Np,fold0,LEGIdata)

    % plot the change of E,I,R curves wrt time
    Ntime = length(TLEGI);
    Efront = zeros(Ntime,1);
    Ifront = zeros(Ntime,1);
    Rfront = zeros(Ntime,1);
    Eback = zeros(Ntime,1);
    Iback = zeros(Ntime,1);
    Rback = zeros(Ntime,1);
    Emax = 0;
    Emin = 0;
    Imax = 0;
    Imin = 0;
    Rmax = RInit;
    Rmin = RInit;

    for i = 1:Ntime
        Ecurrent = YLEGI(i,1:LEGIparam.Np);
        if min(Ecurrent) < Emin
            Emin = min(Ecurrent);
        end
        if max(Ecurrent) > Emax
            Emax = max(Ecurrent);
        end
        Efront(i) = Ecurrent(1);
        Eback(i) = Ecurrent(round(Np/2));

        Icurrent = YLEGI(i,LEGIparam.Np+1:2*LEGIparam.Np);
        if min(Icurrent) < Imin
            Imin = min(Icurrent);
        end
        if max(Icurrent) > Imax
            Imax = max(Icurrent);
        end
        Ifront(i) = Icurrent(1);
        Iback(i) = Icurrent(round(Np/2));

        Rcurrent = YLEGI(i,2*LEGIparam.Np+1:end);
        if min(Rcurrent) < Rmin
            Rmin = min(Rcurrent);
        end
        if max(Rcurrent) > Rmax
            Rmax = max(Rcurrent);
        end
        Rfront(i) = Rcurrent(1);
        Rback(i) = Rcurrent(round(Np/2));
    end

    figure
    subplot(2,1,1),plot(TLEGI,Efront,'r',TLEGI,Eback,'r--',TLEGI,Ifront,'b',TLEGI,Iback,'b--'); 
    xlim([TLEGI(1) TLEGI(end)])
    subplot(2,1,2),plot(TLEGI,Rfront,'c',TLEGI,Rback,'c--'); 
    xlim([TLEGI(1) TLEGI(end)])
    saveas(gcf,strcat(savename,'.tif'), 'tiffn')
    close(gcf)
end


% % generate the result movie
% LEGIavi = strcat(savename,'.avi');
% if exist(LEGIavi,'file')
%     delete(LEGIavi)
% end
% % LEGIavi = avifile(LEGIavi,'compression','None','fps',5,...
% %     'colormap',repmat([0:255]'/255,[1 3]));
% LEGIavi = VideoWriter(LEGIavi);
% LEGIavi.FrameRate = 5;
% open(LEGIavi)
% 
% % to draw the cell boundary
% hI = 2*ceil(Rcell)+2*wOut+1;
% wI = hI;
% I0 = uint8(zeros(hI,wI));
% idxbd = hI*(cbd-1)+rbd;
% % specify the intensity range
% Intensitymin = 50;  % corresponds to Rmin
% Intensitymax = 200; % corresponds to Rmax
% % select time points and draw frame by frame
% Tsel = 1:frameRate:Ntime;
% for i = 1:length(Tsel)
%     % get data
%     Ecurrent = YLEGI(1+(i-1)*frameRate,1:LEGIparam.Np);
%     Icurrent = YLEGI(1+(i-1)*frameRate,LEGIparam.Np+1:2*LEGIparam.Np);
%     Rcurrent = YLEGI(1+(i-1)*frameRate,2*LEGIparam.Np+1:end);
%     % plot
%     I = PlotResult1D(Ecurrent,Icurrent,Rcurrent,Emin,Emax,Imin,Imax,Rmin,Rmax,I0,idxbd,Intensitymin,Intensitymax);
%     % add frame to the movie
%     writeVideo(LEGIavi,I);
% end
% 
% close(LEGIavi);

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

