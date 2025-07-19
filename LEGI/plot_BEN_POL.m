function plot_BEN_POL(Y,timestep,maxTime,Np,resultfoldbase)

resultfold = strcat(resultfoldbase,'_0');
% rename if the result folder already exists
if exist(resultfold,'dir')
    resultidx = 1;
    while exist(strcat(resultfoldbase,'_',num2str(resultidx)),'dir')
        resultidx = resultidx+1;
    end
    resultfold = strcat(resultfoldbase,'_',num2str(resultidx));
end
mkdir(resultfold);

% plot kymograph
Yinhibitor = Y(1:ceil(1/timestep):end,1:4:end);    % sampled once every second at all Np points
Yinhibitor = horzcat(Yinhibitor(:,round(end/2):end),Yinhibitor(:,1:round(end/2)));  % align with the axis of kymograph
Ikymo = Yinhibitor';
Ikymo = flipud(Ikymo);  % flip up and down so that the y-axis is lower at bottom

subplot(4,1,1),imshow(Ikymo,[]);
colormap jet
hold on;
xlabel('Inhibitor');
ylabel('Angle');
% label x axis
for xkymo = 0:300:maxTime
    text(xkymo+5,Np+20,num2str(xkymo),'HorizontalAlignment','center');
end
% label y axis
text(-10, 1, '180','HorizontalAlignment','right');
text(-10, (Np+1)/2, '0','HorizontalAlignment','right');
text(-10, Np+1, '-180','HorizontalAlignment','right');

% X1 = [10 10];
% Y1 = [1 314];
% plot(X1,Y1,'w--');
% X2 = [430 430];
% Y2 = [1 314];
% plot(X2,Y2,'w--');


% plot kymograph
Yinhibitor = Y(1:ceil(1/timestep):end,2:4:end);    % sampled once every second at all Np points
Yinhibitor = horzcat(Yinhibitor(:,round(end/2):end),Yinhibitor(:,1:round(end/2)));  % align with the axis of kymograph
Ikymo = Yinhibitor';
Ikymo = flipud(Ikymo);  % flip up and down so that the y-axis is lower at bottom

subplot(4,1,2),imshow(Ikymo,[]);
colormap jet
hold on;
xlabel('Activitor');
ylabel('Angle');
% label x axis
for xkymo = 0:300:maxTime
    text(xkymo+5,Np+20,num2str(xkymo),'HorizontalAlignment','center');
end
% label y axis
text(-10, 1, '180','HorizontalAlignment','right');
text(-10, (Np+1)/2, '0','HorizontalAlignment','right');
text(-10, Np+1, '-180','HorizontalAlignment','right');

% plot kymograph
Yinhibitor = Y(1:ceil(1/timestep):end,3:4:end);    % sampled once every second at all Np points
Yinhibitor = horzcat(Yinhibitor(:,round(end/2):end),Yinhibitor(:,1:round(end/2)));  % align with the axis of kymograph
Ikymo = Yinhibitor';
Ikymo = flipud(Ikymo);  % flip up and down so that the y-axis is lower at bottom

subplot(4,1,3),imshow(Ikymo,[]);
colormap jet
hold on;
xlabel('Z');
ylabel('Angle');
% label x axis
for xkymo = 0:300:maxTime
    text(xkymo+5,Np+20,num2str(xkymo),'HorizontalAlignment','center');
end
% label y axis
text(-10, 1, '180','HorizontalAlignment','right');
text(-10, (Np+1)/2, '0','HorizontalAlignment','right');
text(-10, Np+1, '-180','HorizontalAlignment','right');

% X1 = [10 10];
% Y1 = [1 314];
% plot(X1,Y1,'w--');
% X2 = [430 430];
% Y2 = [1 314];
% plot(X2,Y2,'w--');

% plot kymograph
Yinhibitor = Y(1:ceil(1/timestep):end,4:4:end);    % sampled once every second at all Np points
Yinhibitor = horzcat(Yinhibitor(:,round(end/2):end),Yinhibitor(:,1:round(end/2)));  % align with the axis of kymograph
Ikymo = Yinhibitor';
Ikymo = flipud(Ikymo);  % flip up and down so that the y-axis is lower at bottom

subplot(4,1,4),imshow(Ikymo,[]);
colormap jet
hold on;
xlabel('W');
ylabel('Angle');
% label x axis
for xkymo = 0:300:maxTime
    text(xkymo+5,Np+20,num2str(xkymo),'HorizontalAlignment','center');
end
% label y axis
text(-10, 1, '180','HorizontalAlignment','right');
text(-10, (Np+1)/2, '0','HorizontalAlignment','right');
text(-10, Np+1, '-180','HorizontalAlignment','right');

% save kymograph
saveas(gcf,strcat(resultfold,'/kymograph.eps'), 'epsc')
saveas(gcf,strcat(resultfold,'/kymograph.tif'), 'tiffn')
close(gcf)
end