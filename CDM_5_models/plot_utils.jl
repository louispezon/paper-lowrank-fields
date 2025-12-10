using LinearAlgebra: norm


tab10_cols = [
    [0.121, 0.466, 0.705],
    [1.000, 0.498, 0.054],
    [0.172, 0.627, 0.172],
    [0.839, 0.152, 0.156],
    [0.580, 0.403, 0.741],
    [0.549, 0.337, 0.294],
    [0.890, 0.466, 0.760],
    [0.498, 0.498, 0.498],
    [0.737, 0.741, 0.133],
    [0.090, 0.745, 0.811]
]


function frac_var_explained(axs, act)
    axs = axs ./ norm.(eachcol(axs))'  # normalize axs
    total_var = sum(var(act; dims=2))
    return [sum(var((axs[:,i:i]' * act); dims=2)) / total_var for i in 1:size(axs, 2)]
end


######### colormaps from given rgb color
function cmap(rgb_col; bright=0.7, dark=0.4)
    start = bright*[1, 1, 1] + (1-bright)*rgb_col
    mid = rgb_col
    endpoint = (1-dark)*rgb_col
    matplotlib.colors.LinearSegmentedColormap.from_list("",[endpoint,mid,start])
end

function plot_cmaps(names)
    x = 0:0.01:1
    figure(figsize=(.5*length(names), 2))
    for (i,name) in enumerate(names)
        scatter(0*x.+i, x, c=x, cmap=cmap(cols[name]), s=50, alpha=.9)
    end
    xticks(1:length(names), names)
    yticks([])
    xlim(0.5, length(names)+0.5)
    tight_layout()
    box(false)
end


##########################################
##### Plot neuronal embeddings from PC loadings

# function plot_embeds(embed, ratios; dims=[1,2,3], tit="", kwargs...)
    
#     labels(dims, ratios) = [@sprintf("PC%d (%1.f %%)",dim, 100*ratios[dim]) for dim in dims]

#     sc = scatter3D(embed[:,dims[1]], embed[:,dims[2]], embed[:,dims[3]]; c=embed[:,dims[3]], kwargs...)
#     labs = labels(dims, ratios)
#     xlabel(labs[1]) ; ylabel(labs[2]) ; zlabel(labs[3])
#     # xticks(-1:0.5:1,[]); yticks(-1:0.5:1,[]); zticks(-1:0.5:1,[]); 
#     title(tit)
#     tight_layout()
#     return sc
# end

# function shadow(proj_axes, fig, embeds; dims=[1,2,3], zmin = nothing, kwargs...)
#     figure(fig)
#     for proj_axis in proj_axes
#         _tmp = copy(embeds) ; 
#         if zmin === nothing; zmin = minimum(_tmp[:,dims[proj_axis]])*1.2 ; end
#         _tmp[:,dims[proj_axis]].= zmin ;
#         # plot_3d_loadings(_tmp; dims=dims, c="Gray", alpha=.1)
#         scatter(_tmp[:,dims[1]], _tmp[:,dims[2]], _tmp[:,dims[3]]; 
#             c="Gray", alpha=.1, #s=10,
#             kwargs...)
#     end
# end

function shadow_(ax, proj_axes, embeds; 
    plot_fun=scatter, dims=[1,2,3], zmin=nothing, c="Gray", alpha=.1, kwargs...)
    sca(ax)
    for proj_axis in proj_axes
        _tmp = copy(embeds)
        if zmin === nothing; zmin = minimum(_tmp[:,dims[proj_axis]])*1.2; end
        _tmp[:,dims[proj_axis]] .= zmin
        plot_fun(_tmp[:,dims[1]], _tmp[:,dims[2]], _tmp[:,dims[3]]; c=c, alpha=alpha, zorder=-1, kwargs...)
    end
end

function zoom_(ax, zoom)
    ax.set_xlim(ax.get_xlim().* zoom); ax.set_ylim(ax.get_ylim().* zoom); ax.set_zlim(ax.get_zlim().* zoom);
end


########## Plot the readouts 

normalise(dec) = dec/maximum(abs.(dec))
plot_downsample(step, x,y, args... ; kwargs...) = plot(
x[1:step:end],y[1:step:end], args... ; kwargs...)

function plot_readouts(readouts; readout_times=comp_times, tmax=nothing, I1=12*I1, I2=12*I2)
    col_col = "C8"
    col_mot = "C9"

    fig_readout = figure(figsize=(5,2.5))
    for (name,readout) in readouts
        plot(readout_times, readout.choice'|>normalise, 
        lw=2, color=cols[name], alpha=0.9)#, label=name)
    end

    plot_downsample(10, times, I1, label="color input", color=col_col)
    plot_downsample(10, times, I2, label="motion input", color=col_mot)
    
    # plot_downsample(10, times, sense1/5, label="color input", color=col_col)
    # plot_downsample(10, times, sense2/5, label="color input", color=col_mot)

    vscale=1
    correct_choice = (context.>0).*sense1 + (context.<0).*sense2 
    fill_between(times, -vscale, vscale, where=(context .> 0), alpha=0.2, color=col_col)
    fill_between(times, -vscale, vscale, where=(context .< 0), alpha=0.2, color=col_mot)

    if !isnothing(tmax); xlim(0,tmax); end
    xlabel("time [ms]")
    ylabel("firing rate [a.u.]")

    axhline(0, ls="--", c="k", linewidth=.7)
    tight_layout()
end

#### Plot 2D readout trajs
function plot_readout_traj(name,mean_readout; ax=gca(), arrows_::Bool = false, box_::Bool = false, s=10, alpha=0.7, kwargs...)
    sca(ax)
    scatter(mean_readout.color, mean_readout.motion, 
        c = (1:length(mean_readout.color)), 
            cmap = cmap(cols[name]), 
            # cmap = "viridis",
            # c = cols[name],
            s=s, alpha=alpha, kwargs...)
    title(name)
    
    fr = maximum(abs.([mean_readout.color...,mean_readout.motion...]))*1.1
    
    #### option 1 (arrows)
    if arrows_
        arrow(-fr, 0, 2*fr, 0, head_width=0.05*fr, head_length=0.1*fr, fc="k", ec="k", zorder=-1)
        arrow(0, -fr, 0, 2*fr, head_width=0.05*fr, head_length=0.1*fr, fc="k", ec="k", zorder=-1)
        box(false)
        xticks([]); yticks([])
    end
    
    #### option 2 (box)
    if box_
        box(true)
        fr = arrows_ ? fr*1.15 : fr
        xlim(-fr,fr); ylim(-fr,fr)
        xticks([-fr,0,fr],[]); yticks([-fr,0,fr],[])
    end
    
    tight_layout()
    gca().set_aspect("equal")
end