function [tNow,schemeData,membranePotentialFunction,spring,levelSetConfiguration] = LSM(input,tNow,schemeData,membranePotentialFunction,spring,stepSize,levelSetConfiguration)

g = schemeData.grid;
Steps = size(input,1); % each step is timestep seconds
MAX_ALLOWABLE_GRAD_PHI = 3; 
integratorFunc = levelSetConfiguration.integratorFunc;
integratorOptions  = levelSetConfiguration.integratorOptions;

for stepi = 1:Steps
    % How far to step?
    tSpan = [ tNow, tNow + stepSize/100];
   
    % Data to be dealt with during the simualtion
    MPAData = [membranePotentialFunction(:) spring(:)];

    % the externalSignal
    schemeData.externalSignal = input(stepi,:);
     
    % Run the simualtion
    [ tNext, MPAData ] = feval(integratorFunc, @termChemoattractant5,...
        tSpan, MPAData, integratorOptions, schemeData);
    
    % Reinit membraneData if necessary
    phi = reshape(MPAData(:,1), g.shape);
    L = reshape(MPAData(:,2), g.shape);
        
    myNormGradPhi = computeNormOfGradPhi(g, phi, schemeData, -1);

    myNorm = max(myNormGradPhi(:));
    if myNorm > 20
        %When spikes happen, re-calculate phi to be the signed distance
        %funciton based only on the membrane.
        fprintf('\nNorm of grad phi is %g, recreating signed distance function...', myNorm);
        membranePoints = contourc(g.xs{1}(:,1),g.xs{2}(1,:), ...
            phi', [0 0]);
        disp(membranePoints(2,1))
        membranePoints = membranePoints(:,2:(membranePoints(2,1)));
        phi = createSignedDistanceFunction(levelSetConfiguration, ...
            membranePoints);
        fprintf('Done!');
    end
    
    myNormGradPhi = computeNormOfGradPhi(g, phi, schemeData, -1);
    myNorm = max(myNormGradPhi(:));

    while(myNorm > MAX_ALLOWABLE_GRAD_PHI)
        fprintf('\nNorm of grad phi is %g, calling Reinit...', myNorm);
        [phi, ~] = reinitOnTheFly(g, phi, 0.2, 'low');
        myNormGradPhi = computeNormOfGradPhi(g, phi, schemeData, -1);
        myNorm = max(myNormGradPhi(:));
        fprintf('Done!');
    end

    % Get back the correctly shaped data array
    membranePotentialFunction = phi;
    
    % Update membranePoints with new boundary
    membranePoints = contourc(g.xs{1}(:,1),g.xs{2}(1,:), ...
        membranePotentialFunction', [0 0]);
    
    % Pulls off the first 0 contour (in case there are multiple ones...
    %   ie. a cell splitting...)  
    membranePoints = membranePoints(:,2:(membranePoints(2,1)));
    if (polygon_area_2d(membranePoints)>=0)  %if curve is ccw, make it clockwise
    membranePoints = membranePoints(:,end:-1:1);
    end
    % Update the spring potential function
    memSpring = interp2(g.xs{1}', g.xs{2}', L',...
        membranePoints(1,:), membranePoints(2,:));

    memSpring(memSpring > 5) = 5;
    memSpring(memSpring < -5) = -5;
    spring = griddata(membranePoints(1,:), membranePoints(2,:), ...
        memSpring, g.xs{1}, g.xs{2}, 'nearest');
%    spring = h.Feval('griddata',1,membranePoints(1,:), membranePoints(2,:), ...
%         memSpring, g.xs{1}, g.xs{2}, 'nearest');%flori
%    spring = spring{1};%flori
   
   % Update the time step
    tNow = tNext;
    
    % Variables for length of protrusion versus time
    [geom, ~, ~] = polygeom(membranePoints(1,:)',membranePoints(2,:)');
    n = schemeData.n;
    schemeData.membranePoints{n} = membranePoints;
    schemeData.memSpring{n} = memSpring;
    schemeData.surfArea(n) = geom(1);
    schemeData.perimeter(n) = geom(4);
    schemeData.centerX(n) = geom(2);
    schemeData.centerY(n) = geom(3);
    schemeData.time(n) = tNow;
    schemeData.n = n+1;
    
%     fprintf('\n cellcenter = [%g,%g]',geom(2),geom(3))

    % Finish this time step
%     if mod(stepi,600) == 0
%         fprintf('\nJust finished time point %g seconds, computation time %g seconds',...
%             tNow*timeunit, cputime-tStart);
%     end
    % Print the area within the time line
%     figure(2); clf;
%     membraneToCenterX = membranePoints(1,:) - geom(2);
%     membraneToCenterY = membranePoints(2,:) - geom(3);
%     angleToCenter = atan2(membraneToCenterY,membraneToCenterX);
%     externalSignal = schemeData.externalSignal;
%     signalAngle = [-pi:(2*pi)/313:pi]; 
%     % To make sure signalAngle lies into range [-pi pi]
%     signalAngle(end) = signalAngle(end) + 0.002;
%     receptorACTIVITY = interp1(signalAngle,externalSignal,angleToCenter,'nearest');
%     scatter(membranePoints(1,:),membranePoints(2,:),...
%         24,receptorACTIVITY,'filled');
%     hold on
%     scatter(geom(2),geom(3),60,'r','filled')
%     hold off
%     midx = mean(membranePoints(1,:));
%     midy = mean(membranePoints(2,:));
%     axis([midx-20 midx+20 midy-20 midy+20]);
%     axis equal;
%     drawnow;

end

end