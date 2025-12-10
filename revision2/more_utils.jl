######### Additional utils for Revision 2

function rep_average_(act_; N_reps, T = nothing)
    if isnothing(T); T = size(act_,2) / N_reps |> round |> Int; end
    @assert size(act_,2) == N_reps * T
    d = size(act_,1)
    act_reps = [hcat([act_[i,(t-1)*T+1:t*T] for t in 1:N_reps]...) for i in 1:d]
    averaged = hcat([mean(act_reps[i], dims=2) for i in 1:d]...)'
    return averaged
end


rep_average_dict(dict, Nreps) = Dict( name => rep_average_(act, N_reps=Nreps) for (name,act) in dict)



function plot_trajs_(pc_trajs::Dict ; 
    plot_fun=plot, 
    shadow_dims = [], axes=nothing, pca_results=pca_results, kwargs_shadow=Dict(:alpha => 0.5), 
    kwargs...)

    if axes === nothing
        fig, axes = subplots(1,length(pc_trajs), figsize=(4*length(pc_trajs),4), subplot_kw = Dict("projection" => "3d"))
    end

    for (ax, (name, traj)) in zip(axes, pc_trajs)
        sca(ax)
        plot_fun(traj[1,:], traj[2,:], traj[3,:]; color=cols[name], kwargs...)
        for dim in shadow_dims
            shadow_(ax, dim, traj'; plot_fun=plot_fun, kwargs_shadow...)
        end
        # title(name)
        ax.set_xticks([0], []); ax.set_yticks([0], []); ax.set_zticks([0], []);
    end
    # suptitle("PC trajectories")
    return fig, axes
end

function display_text(ax, text; x=1, y=0, z=0, kwargs...)
    x = (1-x)*ax.get_xlim()[1] + x*ax.get_xlim()[2]
    y = (1-y)*ax.get_ylim()[1] + y*ax.get_ylim()[2]
    z = (1-z)*ax.get_zlim()[1] + z*ax.get_zlim()[2]
    ax.text(x,y,z, text; kwargs...)
end
