function [conversion_factor] = conc2count(cellradius,nbin)
%computing the conversion factor to go from concentration to count/bin
%assuming a cell height of 3um
%1nM = 0.602 molcules/um^3
% cell radius in unit um, assume cell height = 1.5um
% conversion factor gives the (average) number of molecules for a given membrane
% bin per nM

binvolume = (2*pi*cellradius/nbin)*1.5^2;
conversion_factor = binvolume * 0.602;

end

