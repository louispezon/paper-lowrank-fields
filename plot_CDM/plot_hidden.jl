"""
Plots related to the model 'ring2' with a hidden connectivity component. See Figs. 4 and S7.
"""

###### Load data and perform PCA 
if !@isdefined(pca_results)
    include("plot_CDM.jl")

    interrupt()
end

include("more_utils.jl")
# %% #################################################
"""
Plot PCA loadings vs. ξ
"""

ratios_pca = Dict(
    name => principalvars(pcares)/var(pcares) 
    for (name,pcares) in pca_results
)


models = ["ring",
        # "MS","PS","clust",
        "hid"]

function plot_PC_vs_ξ(models, N_pcs; emb_dict = pc_loadings, a=2, sharey=false, ratios_pca=ratios_pca)
    n_models = length(models)
    _,axs = subplots(n_models,N_pcs, figsize=(.7a*N_pcs,a*n_models), sharex=false, sharey=sharey)
    # figure(figsize=(2*N_pcs,2*n_models), sharex=true, sharey=true)
    for (n,name) in enumerate(models)
        ratios = ratios_pca[name]
        embed = emb_dict[name]*100
        for dim in 1:N_pcs
            ax=axs[n,dim]
            sca(ax)
            if ratios[dim] < 0.005
                box(false) ; xticks([]) ; yticks([])
                continue
            end
            t = @sprintf("PC%d (%1.f %%)",dim, 100*ratios[dim])
            title(t)
            # subplot(n_models,N_pcs, dim + (name-1)*N_pcs, title=t)
            scatter(ξs.+rand(N)*.1, embed[:,dim], alpha=.2 , c=cols[name], marker=".")
            xlim(-1.8,1.8)
            if n==n_models 
                xticks([-1,1])
                xlabel(L"ξ_i") 
            else xticks([])
            end
            (sharey & dim==1) ? yticks([-1,0,1]) : 0#yticks([0,ax.get_yticks()[1]])
        end 
        # t = @sprintf("PC%d (%1.f %%)",dim, 100*ratios[dim])
        # subplot(1,N_pcs,dim, title=t)
        # scatter(ξs, embed[:,dim], alpha=.2)
        tight_layout()
    end
    return axs
end

axs=plot_PC_vs_ξ(models, 6; a=1.6, sharey=false)


# %% ###############################################
"""
Plot membrane potential fluctuations
"""
N_conds = 8
N_reps = 4
T_allconds = N_conds * T_trial

comp_factor = 10
n_steps_allconds = T_allconds/dt / comp_factor |> round |> Int
times_allconds = comp_times[1:n_steps_allconds]

start_times = [i*T_allconds for i in 0:(N_reps-1)]
start_inds = start_times / dt / comp_factor .|> round .|> Int

φex = π/3

models_fluct = ["ring", "hid"]

h_t_trials = Dict()

collect_h_t_trials(h_t; neuron_idx=1:N) = [
        h_t[start_ind+1 : start_ind+n_steps_allconds, neuron_idx] 
        for start_ind in start_inds
]



function plot_h_t_trials(name, ex_angle)
    #### Find index of neuron with angle ex_angle
    Z = hcat(results[name].Z...)
    angles = Z[1,:]
    indx = argmin(abs.(angles .- ex_angle))

    ### Collect trajectories
    h_t_trials = collect_h_t_trials(results[name].h_t, neuron_idx=indx)
    mean_h_t = mean(hcat(h_t_trials...)', dims=1)[1,:]

    for i in 1:N_reps
        # start_ind = start_inds[i]+1
        # end_ind = start_ind + n_steps_allconds
        plot(times_allconds, h_t_trials[i], 
            color=cmap(cols[name])(1-i/N_reps), alpha=.8, lw=1.5)
        end
        # plot(times_allconds, mean_h_t, "k--", lw=1.5)
    
    ylabel(L"$h_i(t)$")
end

ex_angle = π/3

subplots(figsize=(5,3),2,1, sharex=true)

subplot(2,1,1)
name="ring"
plot_h_t_trials(name, ex_angle)

subplot(2,1,2)
name="hid"
plot_h_t_trials(name, ex_angle)

xlabel("time [ms]")

xlim(0,4000)
tight_layout()
