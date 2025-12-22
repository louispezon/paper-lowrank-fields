"""
Limit-cycle trajectories for two models with continuous and discontinuous transfer functions. See Fig. S2F.
"""


using PyPlot
using ProgressBars
pygui(true)

using Random
Random.seed!(4321)

using LinearAlgebra: norm

using ColorSchemes#: phase



include("../src.jl")

# %%
J=1
τ = 10
δ = pi/10

R =  10*1/τ
J = 1

r0 = 2.9348 

N = 2000



## Gauss: σ and J_gaus match derivative at 0 and radius !!
J_gaus = 1.39 * J
f_gaus = (z->z[1],z->z[2])
g_gaus = (z->J_gaus*(rotate(δ)(z))[1] , z->J_gaus*(rotate(δ)(z))[2])

σ =  0.6 #1
Z_gaus = [σ*randn(2) for n in 1:N]

order_Z = sortperm([atan(z[2], z[1]) for z in Z_gaus])

Z_gaus = Z_gaus[order_Z]

### ####
using SpecialFunctions 

φ1(x) = R*(erf(x)+1)/2

φ2(x) = R*(x>0)*(1 - 1/4 * exp(-x))



## ################### run
dt = 2e-1 # ms
T_f = 10000 #ms
times=0:dt:T_f


conn_disorder=false

κ0 = -0.2*r0*[1 -1] #* 0.01

h_t, κ_t, spks = run_ext_input(;
    κ0=κ0, f=f_gaus, g=g_gaus, Z=Z_gaus, φ=φ1, times=times, dt=dt, return_spikes=true)
    
h_t_2, κ_t_2, spks_2 = run_ext_input(;
    κ0=κ0, f=f_gaus, g=g_gaus, Z=Z_gaus, φ=φ2, times=times, dt=dt, return_spikes=true)

## ################### PLOT

function null_formatting(ax; xticks_=[], yticks_=[])
    sca(ax)
    ax.set_xticks(xticks_,[]); ax.set_yticks(yticks_,[])
    box(false) 
    axis("equal")
    tight_layout()
end

col2 = "C2"

# %% ############################################################
#################### plot trajs

fig, ax = subplots(1,1, figsize=(4,4))
traj_inds = 1:10:Int(3000÷dt)
ax.scatter(κ_t[traj_inds,1], κ_t[traj_inds,2], 
    alpha=0.4, s=15,
    c="C1",
    #label=L"\phi_1"
    )#, c=times, cmap="viridis", s=5, alpha=0.7)

ax.scatter(κ_t_2[traj_inds,1], κ_t_2[traj_inds,2], 
    alpha=0.6, s=5,
    c=col2,
    #label=L"\phi_2"
    )#, c=times, cmap="plasma", s=5, alpha=0.7)

null_formatting(ax; xticks_=[0], yticks_=[0])
# 
θs = 0:0.01:2pi
ax.plot(r0*cos.(θs), r0*sin.(θs), c="k", ls="--", lw=2, label="limit cycle")

legend(frameon=true, fontsize=12, loc="upper right")
r0_2 = mean(
    κ_t_2[end-Int(T_f÷(2*dt)):end, :] |> x->sqrt.(x[:,1].^2 .+ x[:,2].^2)
    )
# ax.plot(r0_2*cos.(θs), r0_2*sin.(θs), c="k", ls=":", lw=1)

## ################### tuning curves on the limit cycle
tc(θ,r,z, φ) = φ( r * (cos(θ) * z[1] + sin(θ) * z[2]) )
θs = 0:0.01:2pi
fr_on_lc(r,z, φ) = [tc(θ,r,z, φ) for θ in θs]

n_plot = 19

#### pick among most central neurons
central = sortperm(norm.(Z_gaus))[1:Int(N÷1.6)] |> sort
neurons = central[round.(Int, LinRange(1,length(central),n_plot+1))[1:end-1]]
sample_Z = Z_gaus[neurons]


sample_Z = hcat(sample_Z...)

tuning_curves_1 = [fr_on_lc(r0, z, φ1) for z in eachcol(sample_Z)]
tuning_curves_2 = [fr_on_lc(r0_2, z, φ2) for z in eachcol(sample_Z)]



##################### PLOT tuning curves
angle(z) = atan(z[2], z[1]) / (2pi) .+ 0.5
color(z) = PyPlot.cm.hsv(angle(z))
# colfromrgb(RGBcol) = [RGBcol.r, RGBcol.g, RGBcol.b]
# color(z) = colfromrgb(ColorSchemes.get(phase, angle(z)))
# ColorSchemes.phase.colors
# color([1,0]) # red

function plot_tuning_curves(tc, ax; lw=2, alpha=0.5, kwargs...)
    for i in 1:n_plot
        coli = color(sample_Z[:,i])
        ax.plot(θs, tc[i], alpha=alpha, color=coli, lw=lw, zorder=0, kwargs...)
    end
    ax.set_xticks(0:pi:2pi, [0,"",L"2\pi"])
    ax.set_yticks(0:1, ["0", L"$R$"])
    tight_layout()
end

fig1, ax1 = subplots(1,1, figsize=(3,2)) 
plot_tuning_curves(tuning_curves_1, ax1)
fig2, ax2 = subplots(1,1, figsize=(2.5,1.5))
plot_tuning_curves(tuning_curves_2, ax2, lw=1.5)


###################################### 
# %% where are sampled neurons in circuit space

Zcat = hcat(Z_gaus...)'

figure(figsize=(2,2))
scatter(Zcat[:,1], Zcat[:,2], s=5, c="C2", alpha=.3)#, c=1:N, cmap="hsv", alpha=0.5)
scatter(sample_Z[1,:], sample_Z[2,:], s=20, c=1:n_plot, marker="o", edgecolor="k", cmap="hsv")
scatter(0,0, s=50, c="k", marker="+")

xticks([0],[]); yticks([0],[])
axis("equal")
tight_layout()

xlim([-3*σ 3*σ]...); 
ylim([-3*σ 3*σ]...)


#########
# %% ###################### PCA
LPspks = low_pass_filter(spks')
LPspks_2 = low_pass_filter(spks_2')

using MultivariateStats
print("PCA...")
ndim = 3
M1 = fit(PCA, LPspks; maxoutdim=ndim);
M2 = fit(PCA, LPspks_2; maxoutdim=ndim);
println(" done.")

# %% Embeddings (PC loadings)
embds = loadings(M1)
embds_2 = loadings(M2)

a=4
fig1,ax1 = subplots(1,1, figsize=(a,a))
scatter(embds[:,1], embds[:,2]; c="C1", alpha=0.5, s=15)
null_formatting(ax1)
scatter(embds[neurons,1], embds[neurons,2];  s=40, c=1:n_plot, marker="o", edgecolor="k", cmap="hsv")

fig2,ax2 = subplots(1,1, figsize=(4,4))
scatter(embds_2[:,1], embds_2[:,2]; c=col2, alpha=0.5, s=15)
null_formatting(ax2)
scatter(embds_2[neurons,1], embds_2[neurons,2];  s=40, c=1:n_plot, marker="o", edgecolor="k", cmap="hsv")



# %% Plot transfer functions
function plot_phi(phi, ax; kwargs...)
    sca(ax)
    xs = -2:0.01:2
    plot(xs, phi.(xs); kwargs...)
    ax.set_xticks([0],[""])
    ax.set_yticks(0:1,["0",L"$R$"])
    plot(xs,0*xs, "--", c="k", lw=1, zorder=10, alpha=0.5)
    tight_layout()
end

fig, axs = subplots(1,2, figsize=(2,1))
plot_phi(φ1, axs[1], c="C1", lw=3, alpha=0.9)
plot_phi(φ2, axs[2], c=col2, lw=3, alpha=0.9)
for ax in axs
    sca(ax)
    ax.set_xticks([],[]); ax.set_yticks([],[]); box(false)
end

fig,ax = subplots(1,1, figsize=(1,1))
plot_phi(φ1, ax, c="C1", lw=3, alpha=0.5)
plot_phi(φ2, ax, c=col2, lw=3, alpha=0.9)
ax.set_xticks([],[]); ax.set_yticks([],[]); box(false)