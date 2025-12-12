

###### Load data and perform PCA 
if !@isdefined(pca_results)
    include("plot_CDM.jl")
end
include("plot_utils.jl")
include(sim_path*"setup_input.jl")


# %% ###################################################
"""
Tuning to task variables
"""

function denoise(pc_result, LPspikes)
    rec = reconstruct(pc_result, predict(pc_result, LPspikes)) .+ pc_result.mean
    return rec
end

denoised_act = Dict(
    name => denoise(pc_result, LPspikes[name]) for (name, pc_result) in pca_results
)

function tuning_to_variable(variable, activity; inds_t = 1:length(comp_times))
    var = variable[inds_t]
    act = activity[inds_t,:]
    count_pos_minus_neg = sum(act[findall(var .> 0),:], dims=1) - sum(act[findall(var .< 0),:], dims=1) 
    return count_pos_minus_neg / (length(inds_t)*dt)
end

# %% ##### Extending sensory input during whole context input
var1 = zeros(length(times))
var2 = zeros(length(times))
for i in 2:length(times)
    var1[i] = (context[i] != 0 ? (sense1[i]!=0 ? sense1[i] : var1[i-1]) : 0)
    var2[i] = context[i] != 0 ? (sense2[i]!=0 ? sense2[i] : var2[i-1]) : 0
end
var1 = var1|> compress_array(10)
var2 = var2|> compress_array(10)

tunings_var1 = Dict(
    name => tuning_to_variable(var1,denoised_act[name]') for name in sim_models
)

tunings_var2 = Dict(
    name => tuning_to_variable(var2,denoised_act[name]') for name in sim_models
)

### Plot tunings
c = 2.1
# figure(figsize=(c,c))
n_models = length(sim_models)
_,axs = subplots(1,n_models, figsize=(c*n_models,c), sharex=true, sharey=true)

for (i,name) in enumerate(sim_models)
    sca(axs[i])
    scatter(tunings_var1[name], tunings_var2[name], c=cols[name], alpha=0.3, s=6)
    
    xlabel("tuning to 'color'") 
    if i==1
        ylabel("tuning to 'motion'") 
    end
    title(name)
    xticks([0],""); yticks([0],"");
    gca().set_aspect("equal")
    tight_layout()
end
