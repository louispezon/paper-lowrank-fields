"""
Primary analysis on CDM models. Used for all plotting scripts. 
"""
# %%
include("../src.jl")

using SpecialFunctions
using Printf
using JLD2
using Random

using MultivariateStats
using PyPlot
pygui(true)

sim_path = "../simulate_CDM/"
include(sim_path*"defs.jl")
include(sim_path*"models.jl")


include("plot_utils.jl")


PLOT = false

## time constant for low-pass filtering. α = 1 / τ_lowpass
## original values: α_LP = 5e-2, α_PCA = 5e-3
α_LP = 1 / 50 # (/ms) for display of trajectories
α_PCA = 1 / 50 # (/ms) ## for PCA


## 32 trials
# @time @load "data/good_CDM5models_N3000_32trials_K0.39_seed1234_compressed.jld2" ## GOOD. with K=0.39. Works well!
@time @load "data/CDM5models_N3000_32trials_K0.39_seed1234_compressed.jld2" 

comp_times = LinRange(0, comp_times[end], size(comp_times,1))
# %% ###############################
cols = Dict(
    "ring" => tab10_cols[1],
    "MS" => tab10_cols[2],
    "PS" => tab10_cols[3],
    "clust" => tab10_cols[4],
    "hid" => tab10_cols[5]
)
 


I1 = inputs[:,1]; I2 = inputs[:,2]; Icontext = inputs[:,3]; input_hidden = inputs[:,4]


## list of simulated models
sim_models = ["ring", "MS", "PS", "clust", "hid"]
 #[name for (name,_) in results]
N_models = length(sim_models)

##############################
### plot latent trajs
if PLOT
    figure()
    for name in sim_models
        plot_traj(results[name].κ_t[:,1:2],unit_κ=models[name].unit_κ, label=name, lw=1, alpha=0.9, color=cols[name])
    end
    legend()
    title("Latent trajectories")
end

# %% ###############################

println("Low Pass... ")
LPspikes = Dict( @time  name => low_pass_filter(res.spikes',α=α_LP) for (name,res) in results)


LPex = LPspikes["ring"]
figure()
plot(LPex[1:5,1:1000]', lw=2)
title("Low-passed spikes")
xlabel("time [ms]")



# %% ###############################
### Decoding

struct Readout
    color::Array{Float64,2}
    motion::Array{Float64,2}
    choice::Array{Float64,2}
end

function readout_color_motion_choice(name) 
    Z = results[name].Z
    renorm_factor = R/dt * 1e3
    Fnorm(F) = F / 
        sum(F.^2) / N / models[name].unit_κ  * renorm_factor


    F1 = models[name].f[1].(Z) |> Fnorm
    F2 = models[name].f[2].(Z) |> Fnorm

    proj_axis_col = F1' 
    proj_axis_mot = F2' 
    proj_axis_choice = proj_axis_col + proj_axis_mot

    proj_color = proj_axis_col * LPspikes[name]
    proj_motion = proj_axis_mot * LPspikes[name]
    proj_choice = proj_axis_choice * LPspikes[name]
    return Readout(proj_color, proj_motion, proj_choice)
end

readouts = Dict( @time name => readout_color_motion_choice(name) for name in sim_models)

# %% ###############################
if PLOT
        
    figure(figsize=(3*N_models, 3))
    for (name,readout) in readouts
        subplot(1,N_models, findfirst(==(name), sim_models))
        plot(readout.color', readout.motion', lw=1,
        color=cols[name], 
        # c = comp_times, s=1,
        alpha=0.8)
        title(name)
        xlabel("color")
        ylabel("motion")
    end
    tight_layout()

    figure()
    for (name,readout) in readouts
        plot(comp_times, readout.choice', lw=2, color=cols[name], alpha=0.5, label=name)
    end
    xlabel(L"t")
    legend()

end




# interrupt()
# %% ###################################################
####################################################
####################################################
## PCA embeddings and trajs
### Do pca on low-passed spikes, with larger time constant (smaller α)
N_PCs = 7
println("Low Pass... ")
act_pca = Dict( @time  name => low_pass_filter(res.spikes' , α=α_PCA) for (name,res) in results)


println("PCA... ")
pca_results = Dict( @time name => 
        fit(PCA, act; maxoutdim=N_PCs) 
        for (name,act) in act_pca)
println("done.")

# %% 
pc_loadings = Dict( @time name => loadings(pca) for (name,pca) in pca_results)
pc_loadings["hid"][:,4].*= -1

# ratios_pca = Dict(
#     name => principalvars(pcares)/var(pcares) 
#     for (name,pcares) in pca_results
# )

ratios_pca = Dict(
    name => frac_var_explained(pc_loadings[name], LPspikes[name]) for name in sim_models
)

function plot_ratio_pca(ax,ratios_pca, markers, cols, N_PCs)
    # f,ax = subplots(1,1, figsize=(3,3))
    for (name,ratios) in ratios_pca
        ax.scatter(1:length(ratios),100*ratios, c=cols[name], label=name, marker=markers[name], s=50)
        # yscale("log")
    end
    xticks(1:N_PCs)
    xlabel("PC")
    ylabel("variance explained (%)")
    legend()
    tight_layout()
end

if PLOT
    markers = Dict("ring" => "o", "MS" => "v", "hid" => "s", "PS" => "^", "clust" => "D")
    f,ax = subplots(1,1, figsize=(3,3))
    plot_ratio_pca(ax,ratios_pca, markers, cols, N_PCs)




    fig, axs = subplots(1,N_models, figsize=(3*N_models,4), subplot_kw = Dict("projection" => "3d"))
    for (i,(name, loa)) in enumerate(pc_loadings)
        axs[i].scatter(loa[:,1], loa[:,2], loa[:,3], c=loa[:,3], cmap=cmap(cols[name], bright=.8, dark=.8), alpha=0.5, s=10)
        shadow_(axs[i], 3, loa[:,1:3]; plot_fun=axs[i].scatter)
        title(name)
    end
    tight_layout()
end



