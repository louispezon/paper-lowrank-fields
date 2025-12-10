"""
PCA analysis on CDM models. See Fig. S6.
"""

# %%
### Load data and perform PCA 
include("plot_CDM.jl")
# %% #################################################
include("plot_utils.jl")
include(sim_path*"setup_input.jl")
include("more_utils.jl")

using LinearAlgebra: qr, norm



# %% #################################
######### Plot PC trajectories
ordered_names = ["ring", "MS", "PS", "clust", "hid"]

pc_trajs = Dict( name => predict(pca_results[name], LPspikes[name])
for name in ordered_names)
pc_trajs["hid"][3,:] .*= -1  # flip PC3 for hid model

f,axs = plot_trajs_(pc_trajs, lw=1,  shadow_dims=[1,3])#, shadow_dims=[1,3])
[zoom_(ax, 0.9) for ax in axs]
f.suptitle("PCA trajectories")

tight_layout()

# %% ####### Trial-averaged trajs
# cut activity into trials

N_conds = 8
T_allconds = N_conds * T_trial
N_reps = N_trials / N_conds |> round |> Int




f,axs = plot_trajs_(rep_average_dict(pc_trajs, N_reps), lw=2, shadow_dims=[1,3], kwargs_shadow=Dict(:alpha => 0.7))#, shadow_dims=[1,3])

for (name, ax) in zip(keys(pc_trajs), axs)
    ax.set_title(name)
    zoom_(ax, 0.9)
    varexp = @sprintf("(%1.f%%)", (100*sum(ratios_pca[name][1:3])))
    display_text(ax, varexp; x=1.1, alpha=0.7)
end
# name="ring"

f.suptitle("PCA trajectories (trial-averaged)")

tight_layout()


# %% #################################
######## Plot 3d trajs (connectivity-defined axes)

function conn_axes_(name)
    hcat([
    ff.(results[name].Z) |> (x -> x / norm(x))
    for ff in models[name].f[1:3]
    ]... ) .* [1 1 -1]
end


conn_axes_dict = Dict( name => conn_axes_(name) for name in sim_models)

conn_def_trajs = Dict(
    name => conn_axes_dict[name]'*LPspikes[name] 
    for name in sim_models
)

# %% ################################# variance explained
explained_var_conn_axes = Dict(
    name => frac_var_explained(
        conn_axes_dict[name],
        LPspikes[name]
    ) for name in sim_models
)

print("Variance explained by conn-defined axes:\n")
for (name, var) in explained_var_conn_axes
    println(name, ": ", round.(100*var, digits=2), " (tot: ", round(100*sum(var), digits=2), "%)")
end

print("Variance explained by 3 first PCs:\n")
for (name, var) in ratios_pca
    println(name, ": ", round.(100*var[1:3], digits=2), " (tot: ", round(100*sum(var[1:3]), digits=2), "%)")
end

# %% plot trajectories projected on conn-defined axes

_ = plot_trajs_(conn_def_trajs,  shadow_dims=[1,3], lw=1)
suptitle("Conn-defined axes trajectories")
tight_layout()

f,axs = plot_trajs_(rep_average_dict(conn_def_trajs, N_reps), lw=2, shadow_dims=[1,3], kwargs_shadow=Dict(:alpha => 0.7))
for (name, ax) in zip(keys(conn_def_trajs), axs)
    ax.set_title(name)
    zoom_(ax, 0.9)
    varexp = @sprintf(" (%1.f%%)", 100*sum(explained_var_conn_axes[name]))
    display_text(ax, varexp; x=1.1, alpha=0.7)
end
f.suptitle("Conn-defined axes trajectories (trial-averaged)")
tight_layout()

# %% #################################

inds_show = Dict(
    key => [1,2,3] for key in sim_models
    # "ring" => [1,2,3],    "MS" => [1,2,3],    "PS" => [1,2,3],    "clust" => [1,2,3],  "hid" => [1,2,4],
)


pc_axes_dict = Dict( name => pc_loadings[name][:,inds_show[name]] for name in sim_models)

# %% ################################# 3d plot of projection axes



function plot_axes_dict(axs, axes_dict)
    for (ax,(name, loa)) in zip(axs, axes_dict)
        sca(ax)
        ax.scatter(loa[:,1], loa[:,2], loa[:,3], c=loa[:,3], cmap=cmap(cols[name], bright=.8, dark=.8), alpha=0.5, s=10)
        shadow_(ax,[3,1], loa; plot_fun=ax.scatter, alpha=0.2, s=5)
        # shadow_(ax,1, loa; plot_fun=ax.scatter, alpha=0.2, s=4)
        zoom_(ax,0.9)
        ax.set_xticks([0], []); ax.set_yticks([0], []); ax.set_zticks([0], []);
    end
    tight_layout()
end

fig,axs = subplots(1,N_models, figsize=(4*N_models,4), subplot_kw=Dict("projection"=>"3d"))
plot_axes_dict(axs, conn_axes_dict)
suptitle("connectivity-defined axes")

fig,axs = subplots(1,N_models, figsize=(4*N_models,4), subplot_kw=Dict("projection"=>"3d"))
plot_axes_dict(axs, pc_axes_dict)
suptitle("PCA axes")
