function [Y] = BEN_POL(initvalues,TLEGI,YLEGI,RInit,timeRange,param)                         
%
% Bistable reaction-diffusion system (excitable network) (X and Y), 
% with polarity module (Z and W).
% inputs regulated by LEGI with gradient stimulus
% - LEGIdata: LEGI output data
% - maxTime: total simulation time

% load system parameters
load('LEGISysParam.mat');

%% simulate system behavior   

% input
input.Np = param.Np;
input.TimeRange = TLEGI;
input.baseMeanbasal = baseMeanbasal;   % corresponding to basal LEGI response
input.baseSigma = baseSigma;    % base fluctuations: white gaussian noise

% base withough noise
R = YLEGI(:,param.Np*2+1:end);
% code below simulate non-adaptive response
% E = YLEGI(:,1:Np);
% R = 1.25 + E;
input.base = (R - RInit*ones(size(R,1),size(R,2)))*baseMeanslope + baseMeanbasal;
% input.base = R;
% specify timeRange for the solver
% timeRange = 0:timestep:maxTime;

% solve SDE
% initvalues = [inhibitor0; activator0; componentZ0; componentW0];
problem = 'RDS_grdt';
numsim = param.Np;
sdetype = 'Ito';
numdepvars = 4;
Y = SDE_euler_mod(initvalues,problem,timeRange,numdepvars,numsim,sdetype,param,input);

% 
%% postprocessing

% resultfoldbase = strcat(fold0,'/',LEGIdata);
% 
% resultfold = strcat(resultfoldbase,'_0');
% % rename if the result folder already exists
% if exist(resultfold,'dir')
%     resultidx = 1;
%     while exist(strcat(resultfoldbase,'_',num2str(resultidx)),'dir')
%         resultidx = resultidx+1;
%     end
%     resultfold = strcat(resultfoldbase,'_',num2str(resultidx));
% end
% mkdir(resultfold);

%% %%%%%% show the results
% % select frames to display
% Ntime = length(timeRange);  % number of total time points
% Nframestep = frameRate/timestep;    % number of time points between 2 frames
% Tsel = timeRange(1:Nframestep:Ntime);
% Nsel = length(Tsel);
% 
% % the max and min arrays of inhibitor, activator and input over time
% Imax = zeros(1,Nsel);
% Imin = zeros(1,Nsel);
% Amax = zeros(1,Nsel);
% Amin = zeros(1,Nsel);
% Zmax = zeros(1,Nsel); % component Z
% Zmin = zeros(1,Nsel);
% Wmax = zeros(1,Nsel); % component W
% Wmin = zeros(1,Nsel);
% 
% Itotal = zeros(1,Nsel);
% Atotal = zeros(1,Nsel);
% Ztotal = zeros(1,Nsel); % component Z
% Wtotal = zeros(1,Nsel); % component W
% 
% inhibitor_ = zeros(Nsel,param.Np);
% activator_ = zeros(Nsel,param.Np);
% componentZ_ = zeros(Nsel,param.Np); % component Z
% componentW_ = zeros(Nsel,param.Np); % component W
% 
% for i = 1:Nsel
%     inhibitor = Y(1+(i-1)*Nframestep,1:4:4*param.Np);
%     activator = Y(1+(i-1)*Nframestep,2:4:4*param.Np);
%     componentZ = Y(1+(i-1)*Nframestep,3:4:4*param.Np);
%     componentW = Y(1+(i-1)*Nframestep,4:4:4*param.Np);
%     
%     inhibitor_(i,:) = inhibitor;
%     activator_(i,:) = activator;
%     componentZ_(i,:) = componentZ;
%     componentW_(i,:) = componentW;
%     % get the max and min arrays
%     % inhibitor
%     Imax(i) = max(inhibitor);
%     Imin(i) = min(inhibitor);
%     % activator
%     Amax(i) = max(activator(:));
%     Amin(i) = min(activator(:));
%     % componentZ
%     Zmax(i) = max(componentZ(:));
%     Zmin(i) = min(componentZ(:));
%     % componentW
%     Wmax(i) = max(componentW(:));
%     Wmin(i) = min(componentW(:));
%     % compute the sum of signals in each frame
%     Itotal(i) = sum(inhibitor(:));
%     Atotal(i) = sum(activator(:));
%     Ztotal(i) = sum(componentZ(:));
%     Wtotal(i) = sum(componentW(:));
% end


% figure,plot(Tsel,Amax,'r', Tsel,Amin,'r--', Tsel,Imax,'b', Tsel,Imin,'b--')
% xlim([0 maxTime])
% title('Variable Max and Min')
% saveas(gcf,strcat(resultfold,'/variablewatch.tif'), 'tiffn')
% close(gcf)
% 
% figure,plot(Tsel,Itotal,'b', Tsel,Atotal,'r')
% xlim([0 maxTime])
% title('Total Signals')
% saveas(gcf,strcat(resultfold,'/total.tif'), 'tiffn')
% close(gcf)

% % plot kymograph
% Yinhibitor = Y(1:ceil(1/timestep):end,1:4:end);    % sampled once every second at all Np points
% Yinhibitor = horzcat(Yinhibitor(:,round(end/2):end),Yinhibitor(:,1:round(end/2)));  % align with the axis of kymograph
% Ikymo = Yinhibitor';
% Ikymo = flipud(Ikymo);  % flip up and down so that the y-axis is lower at bottom
% 
% subplot(4,1,1),imshow(Ikymo,[]);
% colormap jet
% hold on;
% xlabel('Inhibitor');
% ylabel('Angle');
% % label x axis
% for xkymo = 0:300:maxTime
%     text(xkymo+5,Np+20,num2str(xkymo),'HorizontalAlignment','center');
% end
% % label y axis
% text(-10, 1, '180','HorizontalAlignment','right');
% text(-10, (Np+1)/2, '0','HorizontalAlignment','right');
% text(-10, Np+1, '-180','HorizontalAlignment','right');

% X1 = [10 10];
% Y1 = [1 314];
% plot(X1,Y1,'w--');
% X2 = [430 430];
% Y2 = [1 314];
% plot(X2,Y2,'w--');

% 
% % plot kymograph
% Yinhibitor = Y(1:ceil(1/timestep):end,2:4:end);    % sampled once every second at all Np points
% Yinhibitor = horzcat(Yinhibitor(:,round(end/2):end),Yinhibitor(:,1:round(end/2)));  % align with the axis of kymograph
% Ikymo = Yinhibitor';
% Ikymo = flipud(Ikymo);  % flip up and down so that the y-axis is lower at bottom
% 
% subplot(4,1,2),imshow(Ikymo,[]);
% colormap jet
% hold on;
% xlabel('Activitor');
% ylabel('Angle');
% % label x axis
% for xkymo = 0:300:maxTime
%     text(xkymo+5,Np+20,num2str(xkymo),'HorizontalAlignment','center');
% end
% % label y axis
% text(-10, 1, '180','HorizontalAlignment','right');
% text(-10, (Np+1)/2, '0','HorizontalAlignment','right');
% text(-10, Np+1, '-180','HorizontalAlignment','right');

% % plot kymograph
% Yinhibitor = Y(1:ceil(1/timestep):end,3:4:end);    % sampled once every second at all Np points
% Yinhibitor = horzcat(Yinhibitor(:,round(end/2):end),Yinhibitor(:,1:round(end/2)));  % align with the axis of kymograph
% Ikymo = Yinhibitor';
% Ikymo = flipud(Ikymo);  % flip up and down so that the y-axis is lower at bottom
% 
% subplot(4,1,3),imshow(Ikymo,[]);
% colormap jet
% hold on;
% xlabel('Z');
% ylabel('Angle');
% % label x axis
% for xkymo = 0:300:maxTime
%     text(xkymo+5,Np+20,num2str(xkymo),'HorizontalAlignment','center');
% end
% % label y axis
% text(-10, 1, '180','HorizontalAlignment','right');
% text(-10, (Np+1)/2, '0','HorizontalAlignment','right');
% text(-10, Np+1, '-180','HorizontalAlignment','right');


% % plot kymograph
% Yinhibitor = Y(1:ceil(1/timestep):end,4:4:end);    % sampled once every second at all Np points
% Yinhibitor = horzcat(Yinhibitor(:,round(end/2):end),Yinhibitor(:,1:round(end/2)));  % align with the axis of kymograph
% Ikymo = Yinhibitor';
% Ikymo = flipud(Ikymo);  % flip up and down so that the y-axis is lower at bottom
% 
% subplot(4,1,4),imshow(Ikymo,[]);
% colormap jet
% hold on;
% xlabel('W');
% ylabel('Angle');
% % label x axis
% for xkymo = 0:300:maxTime
%     text(xkymo+5,Np+20,num2str(xkymo),'HorizontalAlignment','center');
% end
% % label y axis
% text(-10, 1, '180','HorizontalAlignment','right');
% text(-10, (Np+1)/2, '0','HorizontalAlignment','right');
% text(-10, Np+1, '-180','HorizontalAlignment','right');
end

% % save kymograph
% saveas(gcf,strcat(resultfold,'/kymograph.eps'), 'epsc')
% saveas(gcf,strcat(resultfold,'/kymograph.tif'), 'tiffn')
% close(gcf)
% 
% % save result arrays 
% savefile = strcat(resultfold,'/results.mat');
% save(savefile,'timestep','maxTime','Tsel','Y','Itotal','Atotal','Ikymo')

% % draw a cell boundary to show the system response
% Rcell = Np/(2*pi);  % cell radius
% % the drawing field
% wOut = 5;
% hI = 2*ceil(Rcell)+2*wOut+1;
% wI = hI;
% I0 = uint8(zeros(hI,wI));
% % center position
% rctr = ceil(Rcell)+wOut+1;
% cctr = ceil(Rcell)+wOut+1;
% % boundary points
% unitagl = 2*pi/Np; 
% rbd = rctr - round(Rcell*sin(0:unitagl:(Np-1)*unitagl));
% cbd = cctr + round(Rcell*cos(0:unitagl:(Np-1)*unitagl));
% idxbd = hI*(cbd-1)+rbd;
% 
% % initiate the result movie
% FHNavi = strcat(resultfold,'/response.avi');
% if exist(FHNavi,'file')
%     delete(FHNavi)
% end
% FHNavi = avifile(FHNavi,'compression','None','fps',5,...
%     'colormap',repmat([0:255]'/255,[1 3]));
% % specify the intensity range
% Intensitymin = 0;  % corresponds to variable min
% Intensitymax = 255; % corresponds to variable max
% % draw frame by frame
% Imin = min(Imin);
% Imax = max(Imax);
% for i = 1:length(Tsel)
%     % get data
%     inhibitor = inhibitor_(i,:);
%     % plot
%     I = PlotInhibitor1D(inhibitor,Imin,Imax,I0,idxbd,Intensitymin,Intensitymax);
%     % add frame to the movie
%     FHNavi = addframe(FHNavi,I);
% end
% 
% FHNavi = close(FHNavi);
% 
% 
% %% Output
% 
% function I = PlotInhibitor1D(Idata,Imin,Imax,I0,idxbd,Intensitymin,Intensitymax)
% 
% [h,w] = size(I0);
% I = uint8(zeros(h,w));
% I(idxbd) = Intensitymin + (Intensitymax-Intensitymin)*(Idata-Imin)/(Imax-Imin);
% 
% I = uint8(conv2(double(I),[1 2 1;2 4 2;1 2 1]/16,'same'));

