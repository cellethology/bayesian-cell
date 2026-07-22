% Export MATLAB reference values for the deterministic building blocks, so the
% Python port can be checked against them exactly (no RNG, no interpolation).
repo = '/Users/jerrywang/Library/CloudStorage/OneDrive-西湖大学/research/bayesian-cell';
addpath(genpath(repo));

ref = struct();

% --- conc2count / ellipsePerimeter ---
ref.cf_6_100  = conc2count(6,100);
ref.cf_10_50  = conc2count(10,50);
ref.ellipse_6_6 = ellipsePerimeter(6,6);
ref.ellipse_2_5 = ellipsePerimeter(2,5);

% --- membrane angles (phi) as built in nextpos.m ---
m = 100;
phi = linspace(pi,-pi+(2*pi)/m,m)';
phi = circshift(flipud(phi),m/2+1); phi(1) = 0;
ref.phi = phi;

% --- hillfun ---
p.kd = 10*conc2count(6,100); p.rtot = 2000; p.N = 100;
p.dt = 1; p.d = 0.01; p.mean_cell_radius = 6;
p.dx = ellipsePerimeter(6,6)/100;
env = linspace(0.1,50,100);
ref.hill = hillfun(env,p);
ref.env = env;

% --- CNMatrix, scalar and vector d ---
[L1,R1] = CNMatrix(p);
ref.L_scalar = L1; ref.R_scalar = R1;
rng(9); dvec = 0.01*(0.1+rand(1,100));
p2 = p; p2.d = dvec;
[L2,R2] = CNMatrix(p2);
ref.L_vector = L2; ref.R_vector = R2; ref.dvec = dvec;

% --- deterministic receptor_output ---
p.noisy = false;
rvec = 0.9*2000/100*ones(1,100);
ref.act = receptor_output(env,rvec,p);
ref.rvec = rvec;

% --- nextpos, deterministic decoders ---
rng(1); act = 0.3*(1+0.5*rand(1,100));
ref.act_test = act;
ref.next_perfect = nextpos([10 5], act, 1, "perfect");

% --- one full update_receptor step (deterministic) ---
p.L = L1; p.R = R1; p.koff = 0.0678; p.h = 0.002;
p.hprop = p.h/mean(receptor_output(env,rvec,p));
res.f = zeros(3,100); res.FC = zeros(3,1); res.a = zeros(3,100); res.kfb = zeros(3,1);
res.f(1,:) = rvec; res.FC(1) = 2000 - sum(rvec);
res.a(1,:) = receptor_output(env,rvec,p);
res2 = update_receptor(p, env, res, 2);
ref.step_f  = res2.f(2,:);
ref.step_FC = res2.FC(2);
ref.step_a  = res2.a(2,:);
ref.step_kfb = res2.kfb(2);
ref.hprop = p.hprop;

save(fullfile(pwd,'unit_reference.mat'),'-struct','ref','-v7');
fprintf('wrote unit_reference.mat with %d fields\n', numel(fieldnames(ref)));
