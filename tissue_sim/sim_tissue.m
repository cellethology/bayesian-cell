function [csol, cbound] = sim_tissue(params,fname)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Code adapted from K.A. Rejniak, PhD, Moffitt Cancer Center & Research Institute%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

arguments
    params
    fname string = 'tissue_env' %name of saved file
end

kecm = params.kecm; %nM^-1 s^-1
kecmoff = params.kecmoff;
 
% units: micron-microgram-second
% cell boundary coordinates
% rad = params.rad;                             % cell radius [microns]
% centers = params.centers;
% Ncent=size(centers,1);              % number of cell centers
rad = 1;               % cell radius [microns]
centers = [-1000,-1000];
Ncent=size(centers,1);          % number of cell centers

[cell,hb]=DefineCellBoundaryPoints(centers,rad); % cell coordinates & point separation
CellNum=size(centers,1);               % number of cells
csolCell=zeros(Ncent,1);             % absorbed csol per cell

% computational domain
xmin=50; 
xmax=250; % x-axis boundaries
ymin=-450; 
ymax=450; % y-axis boundaries

x0=(xmax+xmin)/2; 
y0=(ymax+ymin)/2;              % center of the domain

% time step
dt = params.dt;         % time step [sec]                

% ligand release
Niter = floor(params.tTot/dt);           % final time: 2 minutes [sec] (was 10*60/dt)
releaseT = floor(params.trelease/dt);
releaseC = params.releaseC;     % 7 ≈ 20pg/hr (Wang and Irvine, 2013)       

% diffusion coefficient
Diff = params.Diff;         % Stability condition: Diff*dt/hg^2<0.25            

% ligand uptake rate by cell
csolUp = params.csolUp;                 % KAR [Units]         

% fluid properties
mu = params.mu;       % fluid viscosity [microgram/(micron*sec)]
hg = params.hg;       % fluid grid width [microns] 
fspeed = params.fspeed;   % parallel fluid flow values [micron/sec] 0.1-2

% ligand decay constants
gammas = params.gammas;
gammab = params.gammab;

% fluid grid & intracellular grid points
uptakeD = params.uptakeD;
Ngx=floor((xmax-xmin)/hg);
Ngy=floor((ymax-ymin)/hg);
xyg=zeros(Ngx+1,Ngy+1);       % grid point coordinates
iscell=zeros(Ngx+1,Ngy+1);    % 0/1 is the grid point inside the cell?
for ii=1:Ngx+1
  for jj=1:Ngy+1
    xyg(ii,jj,1)=xmin+(ii-1)*hg;
    xyg(ii,jj,2)=ymin+(jj-1)*hg;    
   
    dist=pdist2([xyg(ii,jj,1),xyg(ii,jj,2)],centers);
    ij=find(dist<=uptakeD*rad);  % is inside if within uptakeD*rad from cell center
    if length(ij)>0
      iscell(ii,jj)=1;
    end
  end
end
[ixcell,iycell]=find(iscell>0);  % indices of points inside the cells

% coarse grain network into matrix with Ngx x Ngy grid # ECM binding sites
ecmMAT = params.ecmMAT;
ecmsiteC = params.ecmsiteC; % average ecm binding site concentration [nM]
                            % 5nM ≈ one binding site per 0.5um fiber
nrow = size(ecmMAT,1);
ncol = size(ecmMAT,2);
if mod(nrow,2) == 1 || mod(ncol,2) == 1
    error("need ecmMAT to have even valued dimension")
end
% coarse grain by factor of ng, so distance between grid point is ng um
ecmMAT = squeeze(sum(sum(reshape(ecmMAT,hg,nrow/hg,hg,ncol/hg),1),3)); 
fiber = ecmMAT(1:Ngx+1,1:Ngy+1); % taking a subset of 
                                                 % appropriate dimension

% conversion from ecm count to concentration 
normalize_factor = mean(fiber,'all')/ecmsiteC; 
fiber = fiber/normalize_factor; 
% imagesc(fiber')
% pbaspect([10 30 1])
% colorbar()
cbound = zeros(Ngx+1,Ngy+1);

% fluid velocities
uxg=zeros(Ngx+1,Ngy+1);  %% fluid velocity x-coordinate
uyg=zeros(Ngx+1,Ngy+1);  %% fluid velocity y-coordinate
    
% influx boundary velocities 
Nz=floor((ymax-ymin)/hb)-1;
xz(1:Nz,1)=xmin*ones(Nz,1);     % left edge coordinates
xz(1:Nz,2)=[ymin+hb:hb:ymax-hb];
uz=zeros(Nz,2);
uz(1:Nz,1)=fspeed*ones(Nz,1);        % parallel fluid flow values [micron/sec]
  

% zero bundary velocities (top & bottom edge)
Nx=1+floor((xmax-xmin)/hb);
xx=ones(2*Nx,2);
xx(1:Nx,1)=[xmin:hb:xmax];  % bottom edge coordinates
xx(1:Nx,2)=ymin;
xx(Nx+1:2*Nx,1)=[xmin:hb:xmax];
xx(Nx+1:2*Nx,2)=ymax;          % top edge coordinates
ux=zeros(2*Nx,2);              % zero velocities on both edges

% zero cell boundary velocities
Npts=CellNum;
xb(1:Npts,1:2)=cell(1:Npts,1:2); % cell boundaries coordinates
ub=zeros(Npts,2);                % zero velocity on cell boundaries


% draw a figure
h_fig=figure('position',[300,300,750,500]);
set(h_fig,'PaperPositionMode','auto')
set(h_fig,'InvertHardcopy','off')
plot(cell(:,1),cell(:,2),'r.','MarkerSize',10)   % cells
axis([xmin,xmax,ymin,ymax])
axis equal
hold on
plot(xx(:,1),xx(:,2),'c.')   % boundaries
plot(xz(:,1),xz(:,2),'m.')
pause(0.1)

 
% solve forces ff and fb from velocities: uf and ub
  xtot=zeros(Npts+Nz+2*Nx,2); % create one long vector of cell and edge pts
  xtot(1:Npts,1:2)=xb(1:Npts,1:2);
  xtot(Npts+1:Npts+Nz,1:2)=xz(1:Nz,1:2);
  xtot(Npts+Nz+1:end,1:2)=xx(1:2*Nx,1:2);
  
  utot=zeros(Npts+Nz+2*Nx,2); % create one long vector of cell and edge velo
  utot(1:Npts,1:2)=ub(1:Npts,1:2);
  utot(Npts+1:Npts+Nz,1:2)=uz(1:Nz,1:2);
  utot(Npts+Nz+1:end,1:2)=ux(1:2*Nx,1:2);
  
  disp(['total number of points: ',num2str(size(xtot,1))]);
  disp('  ');
    
  % solve regularized Stokelets equations for forces from known velocities  
  disp('calculate forces for a given velocity: '); 
  ftot=ForceFromVelo(xtot,Npts+Nz+2*Nx,xtot,utot,Npts+Nz+2*Nx,mu,hb);
  
  % split forces into separate vectors
  fb(1:Npts,1:2)=ftot(1:Npts,1:2); 
  fz(1:Nz  ,1:2)=ftot(Npts+1:Npts+Nz,1:2); 
  fx(1:2*Nx,1:2)=ftot(Npts+Nz+1:end,1:2); 
    
  
  % solve velocities on the grid from the known forces
  disp('calculate velocities for given forces: '); 
  [uxg,uyg]=VeloFromForce2D(xyg,Ngx+1,Ngy+1,xtot,ftot,Npts+Nz+2*Nx,mu,hb);
  speed=sqrt(uxg.^2+uyg.^2);
  max_speed=max(max(speed))
  pause(0.1)
  
  % solve velocities on the cell boundaries  
  ub=VeloFromForce(xb,Npts,xtot,ftot,Npts+Nz+2*Nx,mu,hb);

  % solve velocities on the domain edge boundaries
  ux=VeloFromForce(xx,2*Nx,xtot,ftot,Npts+Nz+2*Nx,mu,hb);
  
  % solve velocities on the influx domain 
  uz=VeloFromForce(xz,Nz,xtot,ftot,Npts+Nz+2*Nx,mu,hb);
 

  % draw the fluid velocity field
  quiver(xyg(:,:,1),xyg(:,:,2),uxg(:,:),uyg(:,:),1.5,'b') 
 
  % draw the velocities on domain influx  
  quiver(xz(:,1),xz(:,2),uz(:,1),uz(:,2),0.25,'m')
  
  axis([xmin,xmax,ymin,ymax])
  pause(1)
  pbaspect([10 30 1])
  

  disp('draw ligand concentration moving with the fluid flow: ')
  csol=zeros(Ngx+1,Ngy+1);
  starttime = tic;
  for kk=0:Niter
    if (kk<releaseT)
      csol(1:2,1:Ngy+1)=ones(2,Ngy+1)*releaseC; % bolus influx
    else
      csol(1,1:Ngy+1)=0*ones(1,Ngy+1);    % zero influx
    end
    
    % boundary conditions
    csol(1:Ngx+1,1)=csol(1:Ngx+1,2);       
    csol(1:Ngx+1,Ngy+1)=csol(1:Ngx+1,Ngy);
    csol(Ngx+1,1:Ngy+1)=csol(Ngx,1:Ngy+1);
    
    % calculate advection 
    for ii=2:Ngx
      for jj=2:Ngy
        if uxg(ii-1,jj-1)>=0
          csol(ii,jj)=csol(ii,jj)+(dt/hg)*(csol(ii-1,jj)-csol(ii,jj))*uxg(ii-1,jj-1);
        else
          csol(ii,jj)=csol(ii,jj)+(dt/hg)*(csol(ii,jj)-csol(ii+1,jj))*uxg(ii-1,jj-1);
        end
        if uyg(ii-1,jj-1)>=0
          csol(ii,jj)=csol(ii,jj)+(dt/hg)*(csol(ii,jj-1)-csol(ii,jj))*uyg(ii-1,jj-1);
        else
          csol(ii,jj)=csol(ii,jj)+(dt/hg)*(csol(ii,jj)-csol(ii,jj+1))*uyg(ii-1,jj-1);
        end
      end
    end
    
    % calculate diffusion in the interstitium only        
    for ii=2:Ngx
      for jj=2:Ngy
        if iscell(ii,jj)==0  % to avoid crossin cell boundary
          lf=(1-iscell(ii-1,jj));  rg=(1-iscell(ii+1,jj));
          bt=(1-iscell(ii,jj-1));  tp=(1-iscell(ii,jj+1));
          il=lf+rg+tp+bt;             
      
          cent=csol(ii,jj);
          left=(1-iscell(ii-1,jj))*csol(ii-1,jj);
          righ=(1-iscell(ii+1,jj))*csol(ii+1,jj);
          bot =(1-iscell(ii,jj-1))*csol(ii,jj-1);
          top =(1-iscell(ii,jj+1))*csol(ii,jj+1);       
          csol(ii,jj)=cent+(dt*Diff/(hg^2))*(left+righ+top+bot-il*cent);  
        end
      end
    end
    
    % cellular uptake depending on the distance only (1.5*rad)
    for ii=1:Ngx+1
      for jj=1:Ngy+1 
        dist=pdist2([xyg(ii,jj,1),xyg(ii,jj,2)],centers); 
        ll=find(dist<=1.5*rad);
        ll = ll(randperm(length(ll)));
        for mm=1:length(ll)
          if csol(ii,jj)<csolUp*dt
            csolCell(ll(mm),1)=csolCell(ll(mm),1)+csol(ii,jj);
            csol(ii,jj)=0;
          else
            csolCell(ll(mm),1)=csolCell(ll(mm),1)+csolUp*dt;
            csol(ii,jj)=csol(ii,jj)-csolUp*dt;
          end
        end
      end
    end  
    
    % capture by ecm fiber network and enzymatic degradation
    csol = csol - kecm*dt*csol.*(fiber-cbound) - gammas*dt*csol + kecmoff*dt*cbound;
    cbound = cbound + kecm*dt*csol.*(fiber-cbound) - gammab*dt*cbound - kecmoff*dt*cbound;
    
    % drawing a figure
    if mod(kk,100)==0
      clf
      % ligand concentration
      imagesc([xmin:hg:xmax],[ymin:hg:ymax],cbound'+csol')
      caxis([0,max(cbound'+csol',[],'all')]); colorbar; colormap(jet)
      axis([xmin,xmax,ymin,ymax])
      hold on
      % fluid flow
%       quiver(xyg(:,:,1),xyg(:,:,2),uxg(:,:),uyg(:,:),1.5,'w') % fluid flow
      % cell boundary points
      plot(cell(:,1),cell(:,2),'k.','MarkerSize',10)   % cells
      % domain boundary points
      plot(xx(:,1),xx(:,2),'c.')   % boundaries
      plot(xz(:,1),xz(:,2),'m.')

      % cell "nucleus" with color showing the level of absorbed ligand    
%       ind=find(csolCell<=1e-7);
%       plot(centers(ind,1),centers(ind,2),'rd','markerfacecolor','k','MarkerSize',7)   % cells
%       ind=find((csolCell>1e-7)&(csolCell<=0.5));
%       plot(centers(ind,1),centers(ind,2),'mo','markerfacecolor','m','MarkerSize',7)   % cells
%       ind=find((csolCell>0.5)&(csolCell<=5));
%       plot(centers(ind,1),centers(ind,2),'gs','markerfacecolor','g','MarkerSize',7)   % cells
%       ind=find((csolCell>5)&(csolCell<=10));
%       plot(centers(ind,1),centers(ind,2),'y^','markerfacecolor','y','MarkerSize',7)   % cells
%       ind=find(csolCell>10);
%       plot(centers(ind,1),centers(ind,2),'cp','markerfacecolor','c','MarkerSize',7)   % cells

      axis([xmin,xmax,ymin,ymax])
      axis equal
      axis([xmin,xmax,ymin,ymax])
      hold on

      title(strcat('iter=',num2str(kk),' of max iter=',num2str(Niter)),...
          'FontSize',14)
      if (kk==1) pause(1); else pause(0.01); end
    end
  end
  
  figure(1)
  imagesc([xmin:hg:xmax],[ymin:hg:ymax],cbound'+csol');
  colorbar()
  plot(cell(:,1),cell(:,2),'r.','MarkerSize',rad)   % cells
  tEnd = toc(starttime);
  
  save(fname,'cbound','csol','fiber','xmin','xmax','ymin','ymax',...
      'cell','hb','Niter','releaseT','params');
  
end    % end program


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function ForceFromVelo                                              %
% calculates forces at points yy of size Ny that will generate given  %
% velocities ux at points xx; calculates the regularized Stokeslets   %
% transition matrix MM, uses Matlab routines for the LU decomposition % 
% and the gmres method.                                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function fy=ForceFromVelo(yy,Ny,xx,ux,Nx,mu,eps)
  fy=zeros(Ny,2); fyy=zeros(2*Ny,1);
  MM=zeros(2*Ny,2*Ny);
  uy=zeros(2*Ny,1); uy(1:2:2*Ny,1)=ux(1:Nx,1); uy(2:2:2*Ny,1)=ux(1:Nx,2);
  
  W=1/(4.0*pi*mu);
  ep2=eps*eps;
  for ii=1:Ny
    for jj=1:Nx
      dx=yy(ii,1)-xx(jj,1); dy=yy(ii,2)-xx(jj,2);
      r2=dx*dx+dy*dy;
      Hep=0.5*log(r2+ep2)-(ep2/(r2+ep2)); Jep=1/(r2+ep2);
      AA=W*[Jep*dx*dx-Hep,Jep*dx*dy;Jep*dx*dy,Jep*dy*dy-Hep];
      MM(2*ii-1:2*ii,2*jj-1:2*jj)=AA;
    end
  end

  [L1,U1] = lu(MM);
  fyy=gmres(MM,uy,5,1e-5,10,L1,U1);    
  fy(1:end,1)=fyy(1:2:end,1); fy(1:end,2)=fyy(2:2:end,1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function vx=VeloFromForce                                            %
% calculates velocities at points xx of size Nx generated by the given %
% forces fy at points yy; calculates the regularized Stokeslets tran-  %
% sition matrix MM                                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function vx=VeloFromForce(xx,Nx,yy,fy,Ny,mu,eps)
  vx=zeros(Nx,2);

  W=1/(4.0*pi*mu);
  ep2=eps*eps;
  for ii=1:Nx
    for jj=1:Ny
      dx=xx(ii,1)-yy(jj,1); dy=xx(ii,2)-yy(jj,2);
      r2=dx*dx+dy*dy;
      Hep=0.5*log(r2+ep2)-(ep2/(r2+ep2)); Jep=1/(r2+ep2);

      vx(ii,1)=vx(ii,1)-W*Hep*fy(jj,1)+W*dx*Jep*(fy(jj,1)*dx+fy(jj,2)*dy);
      vx(ii,2)=vx(ii,2)-W*Hep*fy(jj,2)+W*dy*Jep*(fy(jj,1)*dx+fy(jj,2)*dy);
    end
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function vx=VeloFromForce2D                                          %
% calculates velocities at points xx of sizes Nx1*Nx2 generated by the %
% given forces fy at points yy; calculates the regularized Stokeslets  %
% transition matrix MM                                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [vx,vy]=VeloFromForce2D(xx,Nx1,Nx2,yy,fy,Ny,mu,eps)
  vx=zeros(Nx1,Nx2); vy=zeros(Nx1,Nx2);
  
  W=1/(4.0*pi*mu);
  ep2=eps*eps;
  for jj=1:Ny
    for ii=1:Nx1
      for kk=1:Nx2  
        dx=xx(ii,kk,1)-yy(jj,1); dy=xx(ii,kk,2)-yy(jj,2);
        r2=dx*dx+dy*dy;
        Hep=0.5*log(r2+ep2)-(ep2/(r2+ep2)); Jep=1/(r2+ep2);

        vx(ii,kk)=vx(ii,kk)-W*Hep*fy(jj,1)+W*dx*Jep*(fy(jj,1)*dx+fy(jj,2)*dy);
        vy(ii,kk)=vy(ii,kk)-W*Hep*fy(jj,2)+W*dy*Jep*(fy(jj,1)*dx+fy(jj,2)*dy);
      end
    end
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function DefineCellBoundaryPoints                           %
% defines the boundary points of all cells                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function centers=DefineCellCenters(nCircles,rad)
  centers = generateCellPos(nCircles,rad);
  centers(:,2)=-centers(:,2);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function DefineCellBoundaryPoints                           %
% defines the boundary points of all cells                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [pts,hb]=DefineCellBoundaryPoints(centers,rad)
%   cells(:,2)=-cells(:,2);

  Ncell=size(centers,1);
  hb=2*pi*rad/25; thn=hb/rad; th=0:thn:2*pi-thn; 
  Npts=0;
  for ii=1:Ncell 
    pts(Npts+1:Npts+length(th),1:2)=[centers(ii,1)+rad*cos(th);...
        centers(ii,2)+rad*sin(th)]'; 
    Npts=Npts+length(th); 
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
