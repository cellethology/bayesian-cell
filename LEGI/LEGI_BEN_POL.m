function [timetaken] = LEGI_BEN_POL(startpt,varargin)
%
% Simulate the cellular gradient sensing responding to spatially gradient 
% of cAMP stimulus using basic LEGI model
% - maxTime: total simulation time
% - cAMPsource: level of cAMP stimulation (> 0); not real concentration
% - ndldist: 10 - 19% gradient; 100 - 6% gradient; gradient is defined as:
%       (cAMP_front - cAMP_back) / [(cAMP_front + cAMP_back) / 2]
% - fold0: fold to save the result

% This script was created by Yuan Xiong and applied in her PNAS paper.
%
% Cleaned by Changji Shi
% 2013/6/19

p = inputParser;
isWithinRange = @(x) (x==1) || (x==2) || (x==3) || (x==4) || (x==5);
addRequired(p,'startpt',@isvector);  % file containing fconc function
% addRequired(p,'param',@isstruct);  % file containing scheme parameters

% optional arguments
addParameter(p, 'signal',1,isWithinRange);
% addParameter(p,'save_data',2,isWithinRange); % 0 = no save, 1 = save without 
%                                              % receptor and env profile
%                                              % 2 = save all data
addParameter(p,'makeplot',false,@islogical);
addParameter(p,'fold0',"",@isstring);
addParameter(p,'LEGIdata',"",@isstring);
addParameter(p,'maxTime',3600,@isnumeric);
addParameter(p,'sumt',100,@isnumeric);

parse(p,startpt,varargin{:});
startpt = p.Results.startpt;
fold0 = p.Results.fold0;
LEGIdata = p.Results.LEGIdata;
makeplot = p.Results.makeplot;
maxTime = p.Results.maxTime;
signal = p.Results.signal;
sumt = p.Results.sumt;

timestep = 0.1;
frameRate = 10;
Np = 314;

% load system parameters
load('/Users/jerrywang/Documents/OneDrive - California Institute of Technology/OptimalSearch/sim-code/LEGI-BEN-POL/SysParam.mat');

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
Rcell = 10;  % cell radius
phi = linspace(-pi,pi-(2*pi)/Np,Np)';
cellboundary = Rcell*[cos(phi),sin(phi)];

% % the drawing field
% wOut = 5;
% % center position
% rctr = ceil(Rcell)+wOut+1;
% cctr = ceil(Rcell)+wOut+1;
% % boundary points
% unitagl = 2*pi/Np; 
% rbd = rctr - round(Rcell*sin(0:unitagl:(Np-1)*unitagl));
% cbd = cctr + round(Rcell*cos(0:unitagl:(Np-1)*unitagl));
% 
% % compute the steady state cAMP concentration across the space
% rNdl = 5;
% rInf = 10e6;  % assumed distance of Infinity where cCAMP concentration is 0
% 
% % assuming the needle is placed to the right of the cell
% rowNdlctr = rctr;
% colNdlctr = cctr+Rcell+ndldist;  % the 1st point is the front

% % row, col distances of all points to the center of the needle
% rtemp = rbd - rowNdlctr;
% ctemp = cbd - colNdlctr;
% distmatrix = rtemp.*rtemp + ctemp.*ctemp;
% LEGIinput.base = cAMPsource*(log(rInf)-0.5*log(distmatrix))/(log(rInf)-log(rNdl));
% grdtStrt = 2*(max(LEGIinput.base) - min(LEGIinput.base))/(max(LEGIinput.base) + min(LEGIinput.base));

% load('/Users/jerrywang/Documents/OneDrive - California Institute of Technology/SpatialComp/receptor-code/figure_5/data/kymograph_data.mat',...
%     'results');
% globalmu = mean(mean(results.env));
% [nenv, m] = size(results.env);
% results.env = results.env/globalmu * mean(LEGIinput.base);
% results.env = circshift(results.env,m/2-1,2);
% 
% plot(mean(results.env));
% env = zeros(nenv,Np);
% for ii = 1:size(results.env,1)
%     env(ii,:) = interp1(1:m,results.env(ii,:),linspace(1,m,Np));
% end
% LEGIinput.base = repelem(env,round(scheme_param.dt/timestep),1);

%initial conditions
R0 = RInit*ones(1,Np);
I0 = zeros(1,Np);
E0 = zeros(1,Np);
YLEGI_0 = [E0'; I0'; R0']';

% specify timeRange for the solver
moverate = 30;
stepRange = moverate:moverate:maxTime;
timeRange = 0:timestep:moverate;
tlength = length(timeRange)-1;

% building fconc
fname = "tissue_300by900";
load(fname,'cbound','csol','xmin','xmax','ymin','ymax');
ctot = csol(5:end,:) + cbound(5:end,:);
pos = combvec(linspace(1,xmax-xmin,size(ctot,1)),...
                linspace(1,ymax-ymin,size(ctot,2)))';
fenv = scatteredInterpolant(pos,ctot(:),'natural','linear');
% fconc = @(x,y) fenv([x,y])/globalmu * mean(LEGIinput.base);
fconc = @(x,y) fenv([x,y]) * 3.6858;
% make smooth gradient
% envcoord = combvec(1:60,201:700)';
% env = reshape(fconc(envcoord(:,1),envcoord(:,2)),60,500)';
% f = fit((1:60)',mean(env)','exp1');
% fconc = @(x,y) f(x); 

% cell position
cellp = zeros(length(stepRange)+1,2);
cellp(1,:) = startpt;
stepsz = 1;

% solve
YLEGI = zeros(length(stepRange)*tlength+1,3*LEGIparam.Np);
TLEGI = zeros(1,length(stepRange)*tlength+1);
Y = zeros(length(stepRange)*tlength+1,4*LEGIparam.Np);

% starttime = 100;
for ii = 1:length(stepRange)
    timeRange = (ii-1)*moverate:timestep:ii*moverate;
    coord = cellp(ii,:) + cellboundary;
    LEGIinput.base = arrayfun(fconc,coord(:,1),coord(:,2));
    
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
    
    %decode direction using activity of the inhibitor in BEN
%     dirSignal = mean(Ypart(100:end,3:4:4*Np));
    if signal == 1
        dirSignal = mean(YLEGIpart(end-sumt+1:end,1:LEGIparam.Np)); %E
%         disp('test1')
    elseif signal == 2
%         disp('test2')
        dirSignal = mean(YLEGIpart(end-sumt+1:end,2*LEGIparam.Np+1:end)); %RR
    elseif signal == 3
%          disp('test3')
        dirSignal = mean(Ypart(end-sumt+1:end,1:4:4*param.Np)); %inhibitor Y
    elseif signal == 4
%         disp('test4')
        dirSignal = mean(Ypart(end-sumt+1:end,2:4:4*param.Np)); %activator X
    elseif signal == 5
%         disp('test5')
        dirSignal = mean(Ypart(end-sumt+1:end,3:4:4*param.Np)); %Z
    end
    z1 = sum(cos(phi).* dirSignal');
    z2 = sum(sin(phi).* dirSignal');
    movdir = atan2(z2,z1) + randn(1)*0.1;
    cellp(ii+1,:) = cellp(ii,:) + stepsz*[cos(movdir), sin(movdir)];
    
%     disp(cellp(ii+1,:))
    if cellp(ii+1,1) < 5+Rcell % stop when cell is within 5um of source
%         disp(strcat('Time taken: ',num2str(floor(ii*moverate/60)),' mins'))
        break
    elseif ii == length(stepRange)
%         disp("unfinished")
    end
end
timetaken = ii;

% %% postprocessing
% if ~exist(fold0,'dir')
%     mkdir(fold0);
% end

% % save result
% savename = strcat(fold0,'/grdt_',num2str(maxTime),'_',...
%     num2str(cAMPsource),'_',num2str(ndldist));
% 
% save(strcat(savename,'.mat'),'TLEGI','YLEGI','RInit','Y',...
%     'timestep','maxTime','Np');

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

