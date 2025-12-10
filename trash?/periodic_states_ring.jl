using PyPlot
using ProgressBars
pygui(true)

using Random
Random.seed!(321)



include("../src.jl")


τ = 10
R = 10*1/τ
J = 1
print("τJR = ", τ*J*R)
# step(x) = R*(x > 0) # transfer function

N = 2000

## ring:
δ = π/30

CHOICE = 2

#####################
### CHOOSE THE CONNECTIVITY PROFILE
#####################

##############################################
"""
##### CHOICE 2 (works !!)
4 Fourier modes, with freq 1,... 4
2 stable modes: 2 and 3
“local-like connectivity profile” 
"""
to_tuple(array) = (array...,)
to_ind(x) = x|> floor |>Int
n_modes = 4
D = 2*(n_modes)

if CHOICE == 1
    λs = [1, 0, 0, 1]
end

if CHOICE ==2
    # b = 1; λs = [1, 1+2b, 2+b, 1]/4
    ## OR:
    λs = [1, 3, 3, 1]/4
end

f = [[z->cos(n*z), z->sin(n*z)] for n in 1:n_modes] 
f = vcat(f...) |> to_tuple
g = [(z->J*λs[to_ind((n+1)/2)]*f[n](z+δ)) for n in 1:D] |> to_tuple


#####################
### PLOT the connectivity profile
profile(z) = sum([f[2*n-1](z)*λs[n] for n in 1:n_modes], dims=1)
Z = 0:2π/100:2π
figure(figsize=(4,2))
plot(Z.-π, profile.(Z.-π), lw=2)
plot(Z.-π, 0*Z, "k--")
xlabel(L"z")
xticks([-π,0,π], [L"-\pi", "0", L"\pi"])


##############################################
##############################################
######## Draw neuronal locations
# Z = 2*pi*rand(N) |> sort!
Z = 0:2*pi/N:2*pi*(1-1/N) |> shuffle ### equally spaced positions



#####################
using SpecialFunctions
φ(x) = R*(erf(x)+1)/2
# φ(x) = R*(x>0)
k = 3.09 * cos(δ) ## Solution to k = τJΦ(k)


#####################

dt = 1e-1 # ms
T_f = 2000 #ms
times=0:dt:T_f

function gen_comp(n=1) ; return randn(n)/sqrt(2) ; end
######## Initial conditions:
###### For CHOICE 1:

########## For CHOICE 2:
function generate_init_cond(mode)
    vec = ones(D)*0.05
    vec[2*mode-1] = 1
    return vec'
end

if CHOICE == 1
    κ0s = 0.5*[
        [generate_init_cond(mode) for mode in [1,4]]..., 
        # (generate_init_cond(1)+generate_init_cond(2))/2,
        # generate_init_cond(3)
        ]
end

if CHOICE == 2
    κ0s = 0.5*[
        [generate_init_cond(mode) for mode in [3,2]]..., 
        # (generate_init_cond(2)+generate_init_cond(3))/2,
        # generate_init_cond(1)
        ]
end

#####################
### GENERATE external INPUT to switch between modes
function generate_switch_input(switch_times, duration, switch_modes = nothing)
    if switch_modes == nothing
        switch_modes = (rand(length(switch_times))*D/2 .+ 1 ) .|> to_ind
    end
    duration_ind = duration/dt |> to_ind
    switch_times_ind = switch_times/dt .|> to_ind
    input = zeros(length(times),D)
    for (i,switch_ind) in enumerate(switch_times_ind)
        input[switch_ind:switch_ind+duration_ind, 2*switch_modes[i]-1] .= 1
    end
    return input
end

switch_times = [500,1500]
κI = 1/τ * generate_switch_input(switch_times, 50, [2,3]) 

κIs = [nothing,nothing]


h_t_s = []; κ_t_s = []; spk_s = []


for (trial,κ0) in enumerate(κ0s)
    h_t, κ_t, spk  = run_ext_input(
        f=f, g=g, Z=Z, κ0=k*κ0, φ=φ, times=times, dt=dt,
        return_spikes=true,
        κI=κIs[trial]
        )
    push!(h_t_s, h_t)
    push!(κ_t_s, κ_t)
    push!(spk_s, spk)
end


#####################
### PLOT
function plot_traj_Fourier_mode(κ_t; mode=1, unit_κ=k, c0="k", lim = 1.1)
    traj = κ_t[:,2*mode-1:2*mode]/unit_κ
    plot_traj(traj, unit_κ=1, init_state=false, c=c0, alpha=.8)
    ## inital position
    scatter(κ_t[1,2*mode-1]/unit_κ, κ_t[1,2*mode]/unit_κ, c=c0, s=50, marker="*")
    xlim(-lim,lim); ylim(-lim,lim)
    gca().set_aspect("equal")
end

c0s = ["C0", "C1", "C2"]
plot_trajs_Fourier_mode(κ_t_s; mode=1, c0s=c0s) = 
    [plot_traj_Fourier_mode(κ_t; mode=mode, c0=c0) for (κ_t,c0) in zip(κ_t_s,c0s)]

modes = 1:n_modes
figure(figsize=((n_modes)*3,3))
for mode in modes
    PyPlot.subplot(1,n_modes,mode)
    plot_trajs_Fourier_mode(κ_t_s; mode=mode)
    title("Fourier mode #$mode")
    plot_point([0,0], c="k", marker="x")
end
tight_layout()


PyPlot.figure(figsize=(4,2))
for (c,h_t) in zip(c0s, h_t_s)
    scatter(Z, h_t[end,:].|> φ, color=c, s=5)
end
xlabel(L"z_i")
xticks([0,π,2π], ["0", L"\pi", L"2\pi"])
ylabel(L"r_i(t)")
tight_layout()



interrupt()
#####################
### DO PCA
print("Z-scoring...")
z_scores_s = Z_score.(spk_s) 

using MultivariateStats
using Random

print("PCA...")
ndim = 5
pca_result_1 = fit(PCA, z_scores_s[1]'; maxoutdim=ndim)
loadings_1 = loadings(pca_result_1)'

pca_result_2 = fit(PCA, z_scores_s[2]'; maxoutdim=ndim)
loadings_2 = loadings(pca_result_2)'


traj_1 = z_scores_s[1]*loadings_1' / size(loadings_1,2)
traj_2 = z_scores_s[2]*loadings_2' / size(loadings_2,2)


print(" done.\n")



### PLOT 3D loadings of PCA
plot_embed(emb; kwargs...) = scatter3D(emb[1,:],emb[2,:],emb[3,:]; kwargs...)

c = Z 

figure(figsize=(6,3))
subplot(1,2,1)
plot_embed(loadings_1; c=c, s=10, cmap="hsv")
zlim(-1,1)
subplot(1,2,2)
plot_embed(loadings_2; c=c, s=10, cmap="hsv") 
zlim(-1,1)
xlabel("PC1"); ylabel("PC2"); zlabel("PC3")

###### Plot folded ring:
function plot_folded_ring(embs2d, Z; c)
    scatter3D(embs2d[1,:], embs2d[2,:], Z, c=c, cmap="hsv", s=10)
    zlabel("z")
    xticks([]); yticks([]); 
    zticks([0,2π], ["0", L"2\pi"])
end

figure(figsize=(6,3))
subplot(1,2,1)
plot_folded_ring(loadings_1[1:2,:], Z, c=Z)#c0s[1])
xlabel("PC1"); ylabel("PC2")
subplot(1,2,2)
plot_folded_ring(loadings_2[1:2,:], Z, c=Z)#c0s[2])
xlabel("PC1"); ylabel("PC2")
tight_layout()

### Plot PC traj
subs = 10
figure(figsize=(3,3))
# subplot(1,2,1)
plot(traj_1[1:subs:end,1], traj_1[1:subs:end,2], ".", alpha=.5)
# subplot(1,2,2)
plot(traj_2[1:subs:end,1], -traj_2[1:subs:end,2], ".",alpha=.5, c="C1")
xlabel("PC1"); ylabel("PC2")
gca().set_aspect("equal")
tight_layout()

####################### compute embeddings (positions on the ring)

####### tuning to PC1 & PC2
counts_1 = zeros(N,2)
counts_2 = zeros(N,2)

for dim in 1:2
    times_neg = findall(traj_1[:,dim] .< 0)
    tuning_PC = (sum(spk_s[1], dims=1)  .- 2*sum(spk_s[1][times_neg,:], dims=1)) / length(times) / dt
    counts_1[:,dim] = tuning_PC
    times_neg = findall(traj_2[:,dim] .< 0)
    tuning_PC = (sum(spk_s[2], dims=1)  .- 2*sum(spk_s[2][times_neg,:], dims=1)) / length(times) / dt
    counts_2[:,dim] = tuning_PC
end

s=15

figure(figsize=(8,4))
subplot(121)
scatter(counts_1[:,1],counts_1[:,2], alpha=.4, s=s, c=Z, cmap="hsv")
xlabel("tuning to PC1"); ylabel("tuning to PC2")
tight_layout()
gca().set_aspect("equal")
xticks([]); yticks([]); box(false)

# figure(figsize=(4,4))
subplot(122)
scatter(counts_2[:,1],counts_2[:,2], alpha=.4, s=s, c=Z, cmap="hsv")
xlabel("tuning to PC1"); ylabel("tuning to PC2")
tight_layout()
gca().set_aspect("equal")
xticks([]); yticks([]); box(false)
tight_layout()

### Plot folded ring
figure(figsize=(6,3))
subplot(1,2,1)
plot_folded_ring(counts_1', Z, c=Z)
xlabel("tuning to PC1"); ylabel("tuning to PC2")
subplot(1,2,2)
plot_folded_ring(counts_2', Z, c=Z)
xlabel("tuning to PC1"); ylabel("tuning to PC2")
tight_layout()


function emb_(pos) 
    emb = atan.(pos[:,2], pos[:,1]) 
    emb = mod.( (emb.-emb[1]) , 2*pi)
    if emb[end] < π
        emb = -emb .+ 2π
    end
    return emb
end
embed1 = emb_(counts_1)
embed2 = emb_(counts_2) 

figure(figsize=(4,2))
scatter(Z, embed1, s=10)
scatter(Z, embed2, s=10, alpha=.5)
xlabel(L"true location $z_i$") ; ylabel(L"inferred location $\hat z_i$")
xticks([0,2π], ["0", L"2\pi"])
yticks([0,2π], ["0", L"2\pi"])
tight_layout()


########################## Raster plot
spks_rast = spk_s[2]

tshow = T_f#min(1000, T_f)#times[end]/15
ind_tshow = tshow/dt |> to_ind
sp = N/50 |> to_ind

inds = sortperm( Z )
inds_show = inds[1:sp:N]

figure(figsize=(5,2.5))
raster_plot(spks_rast,ind_tshow, inds_show, color="k", lw=.5)
