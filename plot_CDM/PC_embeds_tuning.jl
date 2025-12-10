"""
PC embeddings and tuning in 5 CDM models. See Fig. 5 and Fig. S5.
"""


###### Load data and perform PCA 
@time include("plot_CDM.jl")
include("plot_utils.jl")

# interrupt()

# %% ##################################################
"""
Plot all readouts
"""


include(sim_path*"setup_input.jl")



plot_readouts(readouts)#; tmax=4000)




# %% ###################################################
""" 
Plot all PC embeds
"""


show_dims = Dict(
    "ring" => [1,2,3], 
    "MS" => [1,2,3],
    "hid" => [1,2,4],
    "PS" => [1,2,3],
    "clust" => [1,2,3]
)
shadow_dims =Dict(
    "ring" => 1, 
    "MS" => 1,
    "hid" => 1,
    "PS" => 1,
    "clust" => 1
)


# %%

labels(dims, ratios) = [@sprintf("PC%d (%1.f %%)",dim, 100*ratios[dim]) for dim in dims]

emb_figs=[]
emb_axs=[]
    
for name in sim_models#[1:1]
    f,ax = subplots(1,1, subplot_kw=Dict("projection"=>"3d"), figsize=(4,4))
    push!(emb_figs, f); push!(emb_axs, ax);
    
    embeds = pc_loadings[name][:,show_dims[name]]
    ax.scatter(embeds[:,1], embeds[:,2], embeds[:,3]; c=embeds[:,3], cmap=cmap(cols[name]; dark=0.8, bright=0.8), s=15, alpha=0.5)

    s_shad = 10
    shadow_(ax, 3, embeds; plot_fun=ax.scatter, s=s_shad)
    shadow_(ax, shadow_dims[name], embeds; plot_fun=ax.scatter, s=s_shad)

    # ax.set_title(name)
    zoom = 0.85
    zoom_(ax, zoom)

    ## uncomment to label axes
    # labs = labels(show_dims[name], ratios_pca[name]) ; ax.set_xlabel(labs[1]) ; ax.set_ylabel(labs[2]) ; ax.set_zlabel(labs[3])
    ## remove ticks:
    ax.set_xticks([-0], []); ax.set_yticks([-0], []); ax.set_zticks([-0], []);
    tight_layout()
end



# %%## PC ratios
markers = Dict("ring" => "o", "MS" => "v", "hid" => "s", "PS" => "^", "clust" => "D")
figure(figsize=(2,3))
for (name,ratios) in ratios_pca
    scatter(1:length(ratios),100*ratios, c=cols[name], label=name, marker=markers[name], s=50)
    # yscale("log")
end
xticks(1:N_PCs)
xlabel("PC")
ylabel("variance explained (%)")
legend()
tight_layout()





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
