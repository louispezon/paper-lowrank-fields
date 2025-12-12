"""
Example: Periodic states in ring model. See Fig. S4.
"""


using PyPlot
using ProgressBars
pygui(true)

using Random
Random.seed!(321)



include("../src.jl")


τ = 10
R = 10*1/τ
J = 3/4
print("τJR = ", τ*J*R)
# step(x) = R*(x > 0) # transfer function

N = 2000

## ring:
δ = π/10

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



function f_g_from_λ(λs; δ=0)
    n_modes = length(λs)
    f = [[z->cos(n*z), z->sin(n*z)] for n in 1:n_modes] 
    f = vcat(f...) |> to_tuple
    g = [(z->J*λs[to_ind((n+1)/2)]*f[n](z+δ/to_ind((n+1)/2))) for n in 1:D] |> to_tuple
    return f, g
end

λs = [1, 3, 3, 1]/3
f, g = f_g_from_λ(λs; δ=δ)

λ0s = [1, 0, 0, 0]
_,g0 = f_g_from_λ(λ0s; δ=δ)

col1 = "C1"
col0 = "C0"


#####################
### PLOT the connectivity profiles
# profile(λs) = z-> J*sum([f[2*n-1](z)*λs[n] for n in 1:n_modes], dims=1)
profile(g) = z-> sum([g[2*n-1](z) for n in 1:n_modes], dims=1) / cos(δ)
function plot_profile(g; kwargs...)
    Z = 0:2π/100:2π
    plot(Z.-π, profile(g).(Z.-π); lw=2, zorder=1, kwargs...)
    # plot(Z.-π, 0*Z, "k--", zorder=0)
    scatter(Z.-π, 0*Z,c=Z, cmap="hsv", zorder=0, s=5)
    xlabel(L"z")
    xticks([-π,0,π], [L"-\pi", "0", L"\pi"])
    tight_layout()
end
# figure(figsize=(6,2))
# _,ax = subplots(1,2, figsize=(6,2), sharey=true)
# subplot(121)

figure(figsize=(2,2))
plot_profile(g; c=col1)
ylim(-1.5,2.5)
yticks([0,1])
# subplot(122)
figure(figsize=(2,2))
plot_profile(g0; c=col0)
ylim(-1.5,1.5)
yticks([0,1])





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

# if CHOICE == 1
#     κ0s = 0.5*[
#         [generate_init_cond(mode) for mode in [1,4]]..., 
#         # (generate_init_cond(1)+generate_init_cond(2))/2,
#         # generate_init_cond(3)
#         ]
# end

# if CHOICE == 2
#     κ0s = 0.5*[
#         [generate_init_cond(mode) for mode in [3,2]]..., 
#         # (generate_init_cond(2)+generate_init_cond(3))/2,
#         # generate_init_cond(1)
#         ]
# end

κ0_per = 0.25*generate_init_cond(3)
κ0_nor = 0.25*generate_init_cond(1)

#######

h_t_per, κ_t_per, spk_per  = run_ext_input(
    f=f, g=g, Z=Z, κ0=k*κ0_per, φ=φ, times=times, dt=dt,
    return_spikes=true
    )
h_t_nor, κ_t_nor, spk_nor  = run_ext_input(
    f=f, g=g0, Z=Z, κ0=k*κ0_nor, φ=φ, times=times, dt=dt,
    return_spikes=true
    )


h_t_s = [h_t_per, h_t_nor]
κ_t_s = [κ_t_per, κ_t_nor]
spk_s = [spk_per, spk_nor]

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
for (i,h_t) in enumerate(h_t_s)
    c = c0s[i]
    scatter(Z, h_t[end,:].|> φ, color=c, s=5)
end
xlabel(L"z_i")
xticks([0,π,2π], ["0", L"\pi", L"2\pi"])
ylabel(L"r_i(t)")
tight_layout()



interrupt()
#####################
### DO PCA
# print("Z-scoring...")
# activity_s = Z_score.(spk_s; return_LPspikes=true) 

print("LP-filtering...")
activity_s = [low_pass_filter(spk'; α=1e-2)' for spk in spk_s]


using MultivariateStats
using Random

print("PCA...")
ndim = 5
pca_result_1 = fit(PCA, activity_s[1]'; maxoutdim=ndim)
loadings_1 = loadings(pca_result_1)'

pca_result_2 = fit(PCA, activity_s[2]'; maxoutdim=ndim)
loadings_2 = loadings(pca_result_2)'


traj_1 = activity_s[1]*loadings_1' / size(loadings_1,2)
traj_2 = activity_s[2]*loadings_2' / size(loadings_2,2)


print(" done.\n")

print("Cumulative variance explained by 2 PCs: ",
    sum(principalvars(pca_result_1)[1:2])/var(pca_result_1)
)
print("Cumulative variance explained by 2 PCs: ",
    sum(principalvars(pca_result_2)[1:2])/var(pca_result_2)
)


### PLOT 3D loadings of PCA
plot_embed(ax, emb; kwargs...) = ax.scatter(emb[1,:],emb[2,:],emb[3,:]; kwargs...)

c = Z 

# figure(figsize=(6,3))
f, axs = subplots(1,2, figsize=(6,3), subplot_kw=Dict("projection"=>"3d"))
plot_embed(axs[1], loadings_1; c=c, s=10, cmap="hsv")
axs[1].set_zlim(-1,1)
plot_embed(axs[2], loadings_2; c=c, s=10, cmap="hsv") 
axs[2].set_zlim(-1,1)
f.suptitle("PC loadings")

###### Plot folded ring:
function plot_folded_ring(embs2d, Z; c)
    scatter3D(embs2d[1,:], embs2d[2,:], Z, c=c, cmap="hsv", s=10)
    zlabel("z")
    xticks([]); yticks([]); 
    zticks([0,2π], ["0", L"2\pi"])
end

figure(figsize=(6,3))
subplot(1,2,1, projection="3d")
plot_folded_ring(loadings_1[1:2,:], Z, c=Z)#c0s[1])
xlabel("PC1"); ylabel("PC2")
subplot(1,2,2, projection="3d")
plot_folded_ring(loadings_2[1:2,:], Z, c=Z)#c0s[2])
xlabel("PC1"); ylabel("PC2")
tight_layout()

### Plot PC traj
subs = 15
ind_end = T_f/3/dt |> to_ind
inds_plot = 1:subs:ind_end
figure(figsize=(2.5,2.5))
# subplot(1,2,1)
scatter(traj_1[inds_plot,1], traj_1[inds_plot,2], alpha=.8, c="C0", s=15)
# subplot(1,2,2)
scatter(traj_2[inds_plot,1], -traj_2[inds_plot,2],alpha=.7, c="C1", s=14)
xlabel("PC1"); ylabel("PC2")
gca().set_aspect("equal")
xticks([]); yticks([]); box(false)
tight_layout()

figure()
plot(times, traj_1[:,1], "C0", lw=2)
plot(times, traj_1[:,2], "C1", lw=2)

####### if PCA done on LP-filtered data
counts_1 = loadings_1[1:2,:]'
counts_2 = loadings_2[1:2,:]'

xylabels() = (xlabel("PC1"); ylabel("PC2")) # xlabel("tuning to PC1"); ylabel("tuning to PC2")

s=15

function plot_loadings(counts)
    scatter(counts[:,1],counts[:,2], alpha=.4, s=s, c=Z, cmap="hsv")
    xylabels()
    gca().set_aspect("equal")
    xticks([]); yticks([]); box(false)
    tight_layout()
end


figure(figsize=(4,4))
# subplot(121)
plot_loadings(counts_1)

figure(figsize=(4,4))
# subplot(122)
plot_loadings(counts_2)

### Plot folded ring
figure(figsize=(6,3))
subplot(1,2,1, projection="3d")
plot_folded_ring(counts_1', Z, c=Z)
xylabels()
tight_layout()
subplot(1,2,2, projection="3d")
plot_folded_ring(counts_2', Z, c=Z)
xylabels()
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


########
