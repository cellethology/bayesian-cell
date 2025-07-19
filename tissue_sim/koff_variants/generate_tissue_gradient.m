%% simulate environment with different ECM binding rate
clear all

load('default_tissue_sim_param','params');
params.kecmoff = 0.05;
params.tTot = 180;
params.releaseC = 1;
params.trelease = params.tTot;
params.fspeed = 0;
params.gammas = 1e-2;

fname = "tissue_env_koff=5e-2";
[csol, cbound] = sim_tissue(params,fname);

%% visualizing ligand landscape
colormap('hot')
files_to_load = ["tissue_env_koff=1e-3", "tissue_env_koff=5e-2", "tissue_env_koff=1e-1"];
n = length(files_to_load);
tiledlayout(1,n)
for ii = 1:n
    load(files_to_load(ii),'cbound','csol','xmin','xmax','ymin','ymax', 'params');
    ctot = csol + cbound;
    pos = combvec(linspace(1,xmax-xmin,size(ctot,1)),...
                    linspace(1,ymax-ymin,size(ctot,2)))';
    fenv = scatteredInterpolant(pos,ctot(:),'natural','linear');
    fconc = @(x,y) fenv([x,y]);
    
    newxmax = 64;
    envcoord = combvec(1:newxmax,1:round(ymax-ymin))';
    env = reshape(fconc(envcoord(:,1),envcoord(:,2)),newxmax,round(ymax-ymin))';
    f = fit((1:newxmax)',mean(env)','exp1');
    fgrad = @(x,y) f(x); %count
    fcount = @(x,y) fconc(x,y);
    
    envcoord = combvec(5:newxmax,101:160)';
    env = reshape(fcount(envcoord(:,1),envcoord(:,2)),newxmax-4,60)';
    nexttile
    imagesc(env);
    title(params.releaseC)
    colorbar
    pbaspect([1,1,1])
end