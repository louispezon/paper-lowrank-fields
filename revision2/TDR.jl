# %%
### Load data and perform PCA 
@time include("../CDM_5_models/plot_CDM.jl")
interrupt()

# %%
using LinearAlgebra: qr, diagm
include("../CDM_5_models/setup_input.jl")

include("../CDM_5_models/plot_utils.jl")
include("more_utils.jl")


# %% #################################
# TDR: Targeted Dimensionality Reduction


# Denoise activity using PCA (used for TDR)

pcaring = pca_results["ring"];

function denoise(pcares, act)
    return reconstruct(pcares,predict(pcares, act))
end


denoised_act = Dict( name => denoise(pca_results[name], act) 
    for (name,act) in LPspikes)

denoised_act_ring = denoised_act["ring"]

neuron=rand(1:N)
figure(figsize=(6,4))
plot(comp_times, denoised_act_ring[neuron,:], lw=2, label="denoised")
plot(comp_times, LPspikes["ring"][neuron,:], lw=1, c="k", alpha=.4)



# %% #################################
### Find axes corresponding to task variables
hcat_ = (x->hcat(x...))

ch_ = readouts["ring"].choice
findchoice(x) = sign.(x[findmax(abs.(x))[2]])
choices = [
    findchoice(ch_[(i-1)*T_trial+1:i*T_trial]) for i in 1:N_trials
    ]

factors_names = ["s1", "s2", "context", "bias"]

F_matrix = hcat([s1, s2, conts, ones(N_trials)]...)' # context, stimulus 1, stimulus 2, choice


function TDR_projector(F, act_)
    N_ = size(act_,1)
    act_trials= [hcat([act_[i,(t-1)*T_trial+1:t*T_trial] for t in 1:N_trials]...) for i in 1:N_]
    
    proj_F = inv( F * F' ) * F
    regression_coeffs = [proj_F * act_trials[i]' for i in 1:N_]

    ### regression coeffs in a d x t x N tensor
    beta_tensor = zeros(size(regression_coeffs[1])..., N_)
    for i in 1:N_
        beta_tensor[:,:,i] = regression_coeffs[i]
    end


    norms_beta = [[beta_tensor[nu,t,:].^2|>sum 
            for t in axes(beta_tensor,2)] for nu in axes(beta_tensor,1)] |> hcat_
    max_ind = [findmax(norms_beta[:,nu])[2] for nu in axes(norms_beta,2)]
    beta_max = [beta_tensor[nu,max_ind[nu],:] for nu in axes(beta_tensor,1)] |> hcat_

    Q,_ = qr(beta_max)
    beta_orth = Matrix(Q)
    return beta_orth
end

tdr_axes = Dict(
    name => TDR_projector(F_matrix, denoised_act[name]) for name in sim_models 
)


# # orient trajs correctly

T1 = 200:300
T2 = 700:800

signs = Dict(
    name => [
        sign(mean( tdr_axes[name][:,1]' * LPspikes[name][:,T2] )),
        sign(mean( tdr_axes[name][:,2]' * LPspikes[name][:,T1] )),
        sign(mean( tdr_axes[name][:,3]' * LPspikes[name][:,T1] )),
        1
    ] for name in sim_models
)


for name in keys(tdr_axes)
    tdr_axes[name] = tdr_axes[name] * diagm(signs[name])
end

tdr_trajs = Dict(
    name => tdr_axes[name]' * LPspikes[name] for name in sim_models
)


# 
N_conds = 8
N_reps = N_trials / N_conds |> round |> Int
avg_tdr_trajs = rep_average_dict(tdr_trajs, N_reps)



# %% #################################
# Fraction of variance explained
explained_var_tdr = Dict(
    name => frac_var_explained(tdr_axes[name], act_pca[name])[1:3] for name in sim_models
)

print("Variance explained by 3 TDR axes:\n")
for (name, var) in explained_var_tdr
    println(name, ": ", round.(100*var, digits=2), " (tot: ", round(100*sum(var), digits=2), "%)")
end

# %% #################################

# %% PLOT

figure(figsize=(12,5))
for (name, traj) in avg_tdr_trajs
    for i in 1:4
        subplot(2,4,i)
        plot(traj[i,:], lw=2, color=cols[name], alpha=.5)
        title(factors_names[i])

        subplot(2,4,i+4)
        plot(F_matrix[i,:],"o:", lw=2)
    end
end
tight_layout()



# %% 

### 3d trajs
f, axs = plot_trajs_(avg_tdr_trajs, lw=2, shadow_dims=[1,3], kwargs_shadow=Dict(:alpha => 0.8))
for (name, ax) in zip(keys(avg_tdr_trajs) , axs)
    ax.set_title(name)
    zoom_(ax, 0.9)
    varexp = @sprintf(" (%1.f%%)", 100*sum(explained_var_tdr[name]))
    display_text(ax, varexp; x=1.1, alpha=0.7)
end

suptitle("TDR trajectories (avg)")
tight_layout()

# %%
f, axs = plot_trajs_(tdr_trajs, lw=1, shadow_dims=[1,3], kwargs_shadow=Dict(:alpha => 0.8))
suptitle("TDR trajectories")
tight_layout()



# #%% Overlaid

# fig,ax = subplots(1,1, subplot_kw=Dict("projection"=>"3d"))
# for name in sim_models
#     par__ = shwn_trajs[name] 
#     ax.plot(par__[1,:], par__[2,:], par__[3,:], color=cols[name], alpha=0.8)
#     # scatter(loa[:,1], loa[:,2], c=loa[:,3], cmap=cmap(cols[name], bright=.8, dark=.8), alpha=0.5, s=5)
#     # colorbar()
#     shadow_(ax, 3, par__'; plot_fun=plot, lw=1, alpha=0.3, zmin=-2, c=.7*cols[name])
#     shadow_(ax, 1, par__'; plot_fun=plot, lw=1, alpha=0.3, zmin=-2, c=.7*cols[name])
#     title("TDR trajs")
# end
# # ax.plot(par__[1,:], par__[2,:], par__[3,:], lw=2, color=cols["ring"], alpha=1)
# # ax.set_xticks([-1,0,1], []); ax.set_yticks([-1,0,1], []); ax.set_zticks([-1,0,1], []);
# tight_layout()

# %% #################################
### 2d trajs (color / motion)

TDR_readouts = Dict(
    name => Readout(traj[1:1,:], traj[2:2,:], 0*traj[3:3,:],
    ) for (name,traj) in avg_tdr_trajs
)

# figure(figsize=(15,3))
# fig, axs = subplots(1,N_models, figsize=(2*N_models,2))#, sharex=true, sharey=true)
for (ax, name ) in zip(axs, sim_models)
    # sca(ax)
    fig,ax = subplots(1,1, figsize=(2,2))
    rd = TDR_readouts[name]
    # ax=subplot(1,N_models, findfirst(==(name), sim_models))
    plot_readout_traj(name,rd; ax=ax, arrows_=true, box_=false, s=10)#, lw=1)
    # ax.plot(rd.color', rd.motion', c=cols[name],lw=1.5)
    # ax.plot(tdr_trajs[name][1,:], tdr_trajs[name][2,:], c="Gray", lw=1, alpha=.5, zorder=-1)#, color='k')
    title(nothing)
    fig.savefig("../figs/revision2/tdr_trajs/"*name*".png", dpi=400, transparent=true)
end

# %% #################################
# Embeddings of the neurons in TDR space



fig,axs = subplots(1,N_models, figsize=(4*N_models,4),
    subplot_kw=Dict("projection"=>"3d"))

for (ax,name) in zip(axs, keys(tdr_axes))
    loa = tdr_axes[name]*sqrt(N)
    sca(ax)
    ax.scatter(loa[:,1], loa[:,2], loa[:,3], c=loa[:,3], cmap=cmap(cols[name], bright=.8, dark=.8), alpha=0.5, s=10)
    # colorbar()
    shadow_(ax,[3,1], loa; plot_fun=ax.scatter, alpha=0.2, s=5)
    # shadow_(ax,1, loa; plot_fun=ax.scatter, alpha=0.2, s=5)
    # title(name)
    # ax.set_xlabel("color"); ax.set_ylabel("motion");
    # # ax.set_aspect("equal")
    ax.set_xticks([0], [])
    ax.set_yticks([0], [])
    ax.set_zticks([0], [])
end
tight_layout()
