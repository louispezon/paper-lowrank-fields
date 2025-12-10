"""
Plot PCA embeddings and neural fields over the plane and the ring, for the models "ring" and "MS".
"""

###### Load data and perform PCA 
include("plot_CDM.jl")

interrupt()
############################################### 
############################################### 
"""
0. Plot population response
"""
plot_names = ["ring","MS"]


ind_plot = 200/dt |> round |> Int
subsample_inds = 1:1:ind_plot
figure(figsize=(3,3))
lw=1
alpha=.5
for name in plot_names
    plot(readouts[name].color[subsample_inds]', readouts[name].motion[subsample_inds]', ".", 
        color=cols[name], alpha=alpha, lw=lw)
end
fr = 0.12
arrow(-fr, 0, 2*fr, 0, head_width=0.05*fr, head_length=0.1*fr, fc="k", ec="k")
arrow(0, -fr, 0, 2*fr, head_width=0.05*fr, head_length=0.1*fr, fc="k", ec="k")
xticks([]); yticks([])
tight_layout()
box(false)

################################################
# readouts
# plot_names = sim_models


include("setup_input.jl")
plot_names = ["ring","MS"]

plot_readouts(
    Dict(name => readouts[name] for name in plot_names);
)

###############################################
###############################################
"""
1. Plot PCA embeddings
"""


a = 0.014
function set_box_lims(xm,ym,zm,a)
    xlim(xm,xm+a); ylim(ym,ym+a); zlim(zm,zm+a); 
    xticks([-xm,0,xm+a],[]); yticks([-ym,0,ym+a],[]); zticks([-zm,0,zm+a],[])
end

name="ring"
f1 = figure(figsize=(5,4))
embeds = plot_pca_embeds(name, s=10)
shadow(3,f1, embeds)
shadow(1,f1, embeds)
set_box_lims(-a,-a,-a,2*a)


name = "MS"
f2 = figure(figsize=(5,4))
embeds = plot_pca_embeds(name)#, s=45)
shadow(3,f2, embeds)
shadow(2,f2, embeds, zmin=a)
set_box_lims(-a,-a,-a,2*a)


[plot_cmaps([name]) for name in plot_names];

############################################
############################################
############################################
"""
2. Structured representation of activity
"""


#############################################
""" 
Get 2D positions of neurons for "MS". 
Obtained by projecting the PC loadings (in 3D) onto a plane defined by an empirical axis.
"""


orthogonal_project(x,axis) = x - axis*(x'*axis)/(axis'*axis)
empirical_axis_MS = [1/5,-1/4,1]

name = "MS"
figure(); 
subplot(1,2,1)
embds = plot_pca_embeds(name)

projected = hcat([orthogonal_project(x,empirical_axis_MS) for x in eachrow(embds)]...)'
scatter3D(projected[:,1],projected[:,2],projected[:,3],c="gray")
subplot(1,2,2)
scatter(projected[:,1],projected[:,2]; 
c = embds[:,3], s=10)
gca().set_aspect("equal")
tight_layout()

##########################################
"""
Plot field over the plane
"""

time_points = [680, 250, 1750] * (1/dt) / 10 .|> round .|> Int
function plot_2d_field(embds,cmap; vmin=0, vmax=1)
    fig2b = PyPlot.figure(figsize=(10,10/3))
    for (i,t) in enumerate(time_points)
        subplot(1,length(time_points),i)
        firing_rates = results["MS"].h_t[t,:] .|> φ
        tripcolor(embds[:,1],embds[:,2],firing_rates, cmap=cmap, alpha=1, lw=0, ec="none", vmin=vmin*R, vmax=vmax*R)
        if i==1
            scatter(embds[:,1],embds[:,2], alpha=1/2, c="orange", s=1) ; 
        end
        ticks=0:0.5: R
        # xticks(ticks, []) ; yticks(ticks, []) #; zticks(ticks, [])
        xticks([]) ; yticks([]) 
        box(false)
    end
    # colorbar(label=L"r(t, z)")
    # colorbar()
    tight_layout()
    # title(cmap)
end



plot_2d_field(projected[:,1:2],"viridis"; vmax=.8, vmin=0.)

##########################################
##########################################
##########################################
"""
Angular positions over the ring.
To get empirical locations uniformly distributed over on the ring:
- sort neurons according to angular position in space of PC1&2
- assign uniform positions in [0,2π] according to sorting 
"""


embds = pc_loadings["ring"][:,1:3]
ang_pos = atan.(embds[:,2], embds[:,1]) 
pos_ring = sortperm(sortperm(ang_pos)) * 2π / N
# plot(ang_pos,pos_ring,".")

Z_ring = hcat(results["ring"].Z...)[1,:]

figure()
subplot(121)
# scatter3D(Zembds_ring[:,1], Zembds_ring[:,2], Zembds_ring[:,3], c=pos_ring, cmap="hsv")
scatter3D(embds[:,1], embds[:,2], embds[:,3], c=pos_ring, cmap="hsv")
subplot(122)
scatter(Z_ring,pos_ring, s=5, alpha=.5)

"""
Plot field over the ring
"""

delay = 250
times_ring = ( [500, 1000, 1500] 
    .+delay) * (1/dt) / 10 .|> round .|> Int
cols_time = ["m", "k", "olive"]
fig1c = PyPlot.figure(figsize=(5,2))
for (i,t) in enumerate(times_ring)
    # subplot(1,length(time_points),i)
    firing_rates = results["ring"].h_t[t,:] .|> φ
    scatter(pos_ring, firing_rates, alpha=.3, color=cols_time[i], s=5)
end
tight_layout()
yticks([0,R], [L"0",L"R"])
xticks([0,2*pi], [L"0",L"2\pi"])






####################################################
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
    "hid" => 2,
    "PS" => 1,
    "clust" => 1
)

for name in sim_models
    f1 = figure(figsize=(5,4))
    embeds = plot_pca_embeds(name; dims=show_dims[name], s=15)

    a = maximum(abs.(embeds))*1.1

    set_box_lims(-a,-a,-a,2*a)

    shadow(3,f1, embeds)
    shadow(shadow_dims[name],f1, embeds, zmin=-a)
    tight_layout()
end

### PC ratios
markers = Dict("ring" => "o", "MS" => "v", "hid" => "s", "PS" => "^", "clust" => "D")
figure(figsize=(4,3))
for (name,ratios) in ratios_pca
    scatter(1:length(ratios),ratios, c=cols[name], label=name, marker=markers[name], s=50)
    # yscale("log")
end
xlabel("PC")
ylabel("variance explained (%)")
legend()
tight_layout()


plot_cmaps(sim_models)