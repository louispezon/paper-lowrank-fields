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
α_LP = 1 / 50 # (/ms) for display of trajectories
α_PCA = 1 / 50 # (/ms) ## for PCA


### Insest here the filename where simulation results are stored
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

N_models = length(sim_models)


# %% ###############################

println("Low Pass... ")
LPspikes = Dict( @time  name => low_pass_filter(res.spikes',α=α_LP) for (name,res) in results)



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


ratios_pca = Dict(
    name => frac_var_explained(pc_loadings[name], LPspikes[name]) for name in sim_models
)


