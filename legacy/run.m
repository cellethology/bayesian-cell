clear all
close all
load('default_param.mat')

%% setting feedback scheme parameter
thour = 4;
scheme_param.mean_cell_radius = 6;
scheme_param.N = 100;
conversion_factor = conc2count(scheme_param.mean_cell_radius,scheme_param.N);
scheme_param.T = 60*60*thour; % seconds
scheme_param.kd = 10*conversion_factor;
scheme_param.rtot = 2000;
scheme_param.receptornoise = 0.1;
scheme_param.temp = 1;
scheme_param.noisy = false;

%% Simulation interstitial cell migration
successrate = racing_cells("tissue_point_noflow", scheme_param,...
        "envmodel","tissue",...
        "task","localization",...
        "source","point",...
        "receptor","feedback",...
        "rununif",false,...
        "save_data",1,...
        "makeplot",false);