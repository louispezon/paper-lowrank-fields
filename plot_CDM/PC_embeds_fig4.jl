"""
PC embeddings in 5 CDM models. See Fig. 4.
"""


###### Load data and perform PCA 
if !@isdefined(pca_results)
    include("plot_CDM.jl")
end
include("plot_utils.jl")





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
figure(figsize=(3,3))
for (name,ratios) in ratios_pca
    scatter(1:length(ratios),100*ratios, c=cols[name], label=name, marker=markers[name], s=50)
    # yscale("log")
end
xticks(1:N_PCs)
xlabel("PC")
ylabel("variance explained (%)")
legend()
tight_layout()


