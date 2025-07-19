%% System equation for the excitable network under gradient stimulus
% revised 3/4/2010 from RDS_nostimulus_sdefile.m to generate IN using
% getRDSInput_BaseMean1D
%
% This script was modified from Yuan's file to accomodate the new Polarity
% module
%
% Changji Shi
% 2011/6/23
%
% Cleaned by Changji Shi
% 2013/6/19

function [out1,out2,out3] = RDS_grdt_sde(t,x,flag,initvalues,SDETYPE,NUMDEPVARS,NUMSIM,param,input)

% SDE model definition: drift, diffusion, derivatives and initial conditions.
%
% [out1,out2,out3] = M9_sdefile(t,x,flag,bigtheta,SDETYPE,NUMDEPVARS,NUMSIM)
%
% IN:     t; working value of independent variable (time)
%         x; working value of dependent variable 
%         flag; a switch, with values 'init' or otherwise
%         bigtheta; complete structural parameter vector
%         SDETYPE; the SDE definition: can be 'Ito' or 'Strat' (Stratonovich)
%         NUMDEPVARS; the number of dependent variables, i.e. the SDE dimension
%         NUMSIM; the number of desired simulations for the SDE numerical integration 
% OUT:    out1; in case of flag='init' is just the initial time, otherwise it is the (vector of) SDE drift(s)
%         out2; in case of flag='init' is the initial value of the dependent variables. Otherwise it is the SDE diffusion(s)
%         out3; in case of flag='init' it is nothing. Otherwise it is the SDE's partial derivative(s) of the diffusion term 

% Copyright (C) 2007, Umberto Picchini  
% umberto.picchini@biomatematica.it
% http://www.biomatematica.it/Pages/Picchini.html
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 2 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.


% Parameters
Xzero1 = initvalues(1,:);     % inhibitor
Xzero2 = initvalues(2,:);     % activator
Xzero3 = initvalues(3,:);     % component Z
Xzero4 = initvalues(4,:);     % component W

IN = getRDSInput_grdt1D_NoNoise(t,input);

if nargin < 3 || isempty(flag)
    
   xsplitted  =  SDE_split_sdeinput(x,NUMDEPVARS);
   
  %::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 
  %::::::::::::::::::::::::::::::::::::::::::  DEFINE HERE THE SDE  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  %:::::::::::::::::::::::: (define the initial conditions at the bottom of the page) ::::::::::::::::::::::::::::::::::::::::::::::::::::
  %:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  X1 = xsplitted{1}; 
  X2 = xsplitted{2};
  X3 = xsplitted{3}; % component Z
  X4 = xsplitted{4}; % component W

   switch upper(SDETYPE)
   case 'ITO'

       % diffusion term
        
         X1shiftleft = [X1(end) X1(1:end-1)];
        X1shiftright = [X1(2:end) X1(1)];
        
        driftX1 = 0.8*(param.Dinhibitor*(X1shiftleft + X1shiftright - 2*X1) ...
            + param.alpha*param.epsilon*(param.gamma*X2 - X1));
        diffusionX1 = 0;
        derivativeX1 = 0; 

        X2shiftleft = [X2(end) X2(1:end-1)];
        X2shiftright = [X2(2:end) X2(1)];
        
        driftX2 = 0.8*(IN + param.Dactivator*(X2shiftleft + X2shiftright - 2*X2) ...
            + param.alpha*((param.a+1)*(X2.^(2*param.n))./(param.a+X2.^(2*param.n)) ...
            - X2 - param.beta*X1) + 2 * (X3 - X4)); %(min(X3,0.15) - X4));
        diffusionX2 = 0.9*1.025 * input.baseSigma;
        derivativeX2 = 0; 
        
        % component Z      
        X3shiftleft = [X3(end) X3(1:end-1)];
        X3shiftright = [X3(2:end) X3(1)];
        
        driftX3 = param.DZ*(X3shiftleft + X3shiftright - 2*X3) + (1.16 * 0.0048 * X2 - 0.0076 * X3); 
        diffusionX3 = 0;
        derivativeX3 = 0; 
        
        % component W
        meanX2 = mean(X2) * ones(size(X2));
        driftX4 = 0.0258 * meanX2 - 0.012 * X4; % was 0.009
        diffusionX4 = 0;
        derivativeX4 = 0;        

   end
   
    out1 = zeros(1,NUMDEPVARS*NUMSIM);
    out1(1:NUMDEPVARS:end) = driftX1;
    out1(2:NUMDEPVARS:end) = driftX2;
    out1(3:NUMDEPVARS:end) = driftX3;
    out1(4:NUMDEPVARS:end) = driftX4;
    
    out2 = zeros(1,NUMDEPVARS*NUMSIM);
    out2(1:NUMDEPVARS:end) = diffusionX1;
    out2(2:NUMDEPVARS:end) = diffusionX2;
    out2(3:NUMDEPVARS:end) = diffusionX3;
    out2(4:NUMDEPVARS:end) = diffusionX4;
    
    out3 = zeros(1,NUMDEPVARS*NUMSIM);
    out3(1:NUMDEPVARS:end) = derivativeX1;
    out3(2:NUMDEPVARS:end) = derivativeX2;
    out3(3:NUMDEPVARS:end) = derivativeX3;
    out3(4:NUMDEPVARS:end) = derivativeX4;

   %:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
   %:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
   %:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    
else
    
    switch(flag)
    case 'init'  
        out1 = t;
        
%:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 
%::::::::::::::::::::::  DEFINE HERE THE SDE INITAL CONDITIONS  :::::::::::::::::::::::::::::::::
%::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

        out2 = [Xzero1;Xzero2;Xzero3;Xzero4];   % write here the SDE initial condition(s)

%:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 
%:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: :::::::::::::::::::::::::::::::::
%::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        
        out3 = [];
        
        
    otherwise
        error(['Unknown flag ''' flag '''.']);
    end
end
