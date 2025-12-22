"""
Limit-cycle trajectories for two different models: compare PC embeddings. See Fig. 3C-D and Fig. S2.
"""

using PyPlot
using ProgressBars
pygui(true)

using Random
Random.seed!(4321)

J=1
τ = 10
δ = pi/10

R =  10*1/τ
J = 1


N = 2000
## ring:
J_ring = J
f_ring=(cos,sin)
g_ring=(z->J_ring*cos(z+δ),z->J_ring*sin(z+δ))
Z_ring = 2*pi*rand(N) |> sort!

r0_ring = 2.9348


## Gauss: σ and J_gaus match derivative at 0 and radius !!
J_gaus = 1.39 * J_ring 
f_gaus = (z->z[1],z->z[2])
g_gaus = (z->J_gaus*(rotate(δ)(z))[1] , z->J_gaus*(rotate(δ)(z))[2])

σ =  0.6 #1
Z_gaus = [σ*randn(2) for n in 1:N]

r0_gaus = r0_ring 


include("../src.jl")
##########
nodyn(x,y) = 0
r(x,y) = sqrt(x^2+y^2)
dkappa(κ) = -κ/τ + J*R/π*rotate(δ)(κ)/r(κ...)
dx(x,y) = dkappa([x,y])[1]
dy(x,y) = dkappa([x,y])[2]



using SpecialFunctions ; φnew(x) = R*(erf(x)+1)/2


dt = 2e-1 # ms
T_f = 10000 #ms
times=0:dt:T_f

conn_disorder=false

## ################### run

κ0 = r0_ring*[1 -1] #* 0.01
h_t_ring, κ_t_ring, κ_th_t_ring, spikes_ring = run_(;κ0,spiking=true,φ=φnew, times=times, f=f_ring, g=g_ring, Z=Z_ring, dx=dx, dy=dy, return_spikes=true, conn_disorder=conn_disorder)
h_t_gaus, κ_t_gaus, κ_th_t_gaus, spikes_gaus = run_(;κ0,spiking=true,φ=φnew, times=times, f=f_gaus, g=g_gaus, Z=Z_gaus, dx=dx, dy=dy,return_spikes=true, conn_disorder=conn_disorder)


## ######################################
############## USELESS to save (runtime ≤ 2min, size of result = 6 Go...)
########################################



function plot_lat_traj(downsample=20)
    plot_traj(κ_t_ring[1:downsample:end,:], unit_κ = r0_ring, marker="o",lw=0, alpha=1, label=L"SNN ($d=1$)")
    plot_traj(κ_t_gaus[1:downsample:end,:], unit_κ = r0_gaus, marker=".",lw=0, alpha=1, label=L"SNN ($d=2$)")

    PyPlot.xlabel(L"\kappa_1 ")
    PyPlot.ylabel(L"\kappa_2 ")
    xticks(-1:1:1) ; yticks(-1:1:1)
    PyPlot.legend(loc="lower left")
    tight_layout()
end

figure(figsize=(4,4))
plot_lat_traj()



interrupt()
## ###################
### DO PCA
z_scores_1d, LPspks_1d = Z_score(spikes_ring, return_LPspikes=true) ;
z_scores_2d, LPspks_2d = Z_score(spikes_gaus, return_LPspikes=true) ; 

# activity_1d = z_scores_1d ; activity_2d = z_scores_2d ; ### do PCA on z-scored spike trains
activity_1d = LPspks_1d'; activity_2d = LPspks_2d'; ### do PCA on LP spikes (no normalisation!) 

using MultivariateStats
using Random


print("PCA...")
ndim = 5
pca_result_ring = fit(PCA, activity_1d'; maxoutdim=ndim)
loadings_ring = loadings(pca_result_ring)
pca_result_gaus = fit(PCA, activity_2d'; maxoutdim=ndim)
loadings_gaus = loadings(pca_result_gaus)
print(" done.\n")



plot_embed(emb; kwargs...) = scatter3D(emb[1,:],emb[2,:],emb[3,:]; kwargs...)
### PLOT 3D loadings of PCA
figure(figsize=(6,3))
subplot(1,2,1, projection="3d")
plot_embed(loadings_ring'; color="C0")
zlim(-1,1)
subplot(1,2,2, projection="3d")
plot_embed(loadings_gaus'; color="C1")
zlim(-1,1)

##### Other PCs are negligible (< 2% variance explained )
ratios_ring = principalvars(pca_result_ring)/var(pca_result_ring)
ratios_gaus = principalvars(pca_result_gaus)/var(pca_result_gaus)
figure(figsize=(3,3))
plot(ratios_ring, "v", lw=0)
plot(ratios_gaus, "^", lw=0)
xlabel("PC index")
ylabel("variance explained")
tight_layout()



print("Explained variance (2PCs): \n",
"ring: ", sum(ratios_ring[1:2]), " % \n",
"gaus: ",sum(ratios_gaus[1:2]),  " % \n")


## ###################
# principalvars(pca_result_ring)
# principalvars(pca_result_gaus)
traj_ring =  activity_1d * loadings_ring ./ principalvars(pca_result_ring)' / N
traj_gaus = activity_2d * loadings_gaus ./ principalvars(pca_result_gaus)' / N


phase_ring = atan.(traj_ring[:,2],traj_ring[:,1]) .+ π
phase_gaus = atan.(traj_gaus[:,2],traj_gaus[:,1]) .+ π

N_cycles_ring = sum( abs.(phase_ring[2:end]-phase_ring[1:end-1]) .> π)
N_cycles_gaus = sum( abs.(phase_gaus[2:end]-phase_gaus[1:end-1]) .> π)


## PLOT 2 FIRST PCs
function plot_2_PCs(subsample=10; rotation=0)
    rot_traj_ring = (traj_ring[1:subsample:end,1:2]' |> rotate(rotation))'
    plot(rot_traj_ring[:,1],rot_traj_ring[:,2], lw=0, marker=".", label="$N_cycles_ring cycles")
    plot(traj_gaus[1:subsample:end,1],traj_gaus[1:subsample:end,2], lw=0, marker=".", label="$N_cycles_gaus cycles", alpha=.5)
    xticks([]); yticks([]); 
    box(false)
    tight_layout()
    plot(rot_traj_ring[1,1],rot_traj_ring[1,2],"*k")#,ms=10)
    plot(traj_gaus[1,1],traj_gaus[1,2],"*k")#,ms=10)
end


## ##########################
f = figure(figsize=(3,3))
# subplot(1,3,1)
rotation = -phase_ring[500]+phase_gaus[500]
plot_2_PCs(rotation=rotation)

gca().set_aspect("equal")


function compute_periods(phase)
    times_cycle = times[findall(abs.(phase[2:end]-phase[1:end-1]) .> π)]
    times_cycle[2:end] .-= times_cycle[1:end-1]
end
periods_count = 20
periods_ring = compute_periods(phase_ring)[end-periods_count:end]
periods_gaus = compute_periods(phase_gaus)[end-periods_count:end]
print("Mean period: \n
    Ring: ", mean(periods_ring), " ± ", std(periods_ring),#/sqrt(periods_count),
    " ms \n
    Gaus: ", mean(periods_gaus), " ± ", std(periods_gaus) ," ms\n"
)




## # Compute 3D embeds
####### tuning to PC1 & PC2
function get_embeds(;is_z_scored=false)
    """ IF PCA was done on z-scored spikes, then we need to compute the TUNING to PCs. 
        IF PCA was done on LP spikes, then we can simply take PC loadings ! """

    if is_z_scored
        counts_ring = zeros(N,3)
        for dim in 1:3
            times_neg = findall( traj_ring[:,dim] .< 0 )
            tuning_PC = (sum(spikes_ring, dims=1)  .- 2*sum(spikes_ring[times_neg,:], dims=1)) / length(times) / dt
            counts_ring[:,dim] = tuning_PC
        end

        counts_gaus = zeros(N,3)
        for dim in 1:3
            times_neg = findall( traj_gaus[:,dim] .< 0 )
            tuning_PC = (sum(spikes_gaus, dims=1)  .- 2*sum(spikes_gaus[times_neg,:], dims=1)) / length(times) / dt
            counts_gaus[:,dim] = tuning_PC
        end
        
    else
        counts_ring = loadings_ring
        counts_gaus = loadings_gaus
    end

    return counts_ring, counts_gaus
end

counts_ring, counts_gaus = get_embeds(;is_z_scored=false)

s=15

figure(figsize=(4,4))
scatter(counts_ring[:,1],counts_ring[:,2], alpha=.4, s=s)
# xlabel("tuning to PC1"); ylabel("tuning to PC2")
gca().set_aspect("equal")
xticks([]); yticks([]); box(false)
tight_layout()

figure(figsize=(4,4))
scatter(counts_gaus[:,1],counts_gaus[:,2], alpha=.5, color="C1", s=s)
# xlabel("tuning to PC1"); ylabel("tuning to PC2")
tight_layout()
gca().set_aspect("equal")
xticks([]); yticks([]); box(false)
tight_layout()

################################################################
########## Plot neural field over the similarity space (ring and plane)
########################################
## ########## empirical locations in similarity space from embeddings

embed_ring = -[atan(pos[2],pos[1]) for pos in eachrow(counts_ring[:,1:2])] .+ π
embed_gaus = rotate(0)(counts_gaus[:,1:2]')'

figure(figsize=(4,2))
subplot(121)
plot(Z_ring,embed_ring)
xlabel(L"z_i") ; ylabel(L"\hat z_i"); tight_layout()

subplot(122)
scatter(embed_gaus[:,1],embed_gaus[:,2], s=5, c="C1")

tight_layout()

## #########
########## Plot field on the ring
########################################


using Colors

fig1c = PyPlot.figure(figsize=(5,2))
Nshow=10
st = 800/Nshow * (1e-1/dt) .|> round .|> Int
t_show = (1:st:st*Nshow .|> Int) .+ 10*st 
fromcoltovec(c) = [c.r,c.g,c.b]
cols = fromcoltovec.(range(colorant"green", stop=colorant"red", length=Nshow))
for (i,t) in enumerate(t_show)
    # subplot(1,length(time_points),i)
    r_ring = h_t_ring[t,:] .|> φnew
    # plot(embed_ring, h_t_ring[t,:] .|> φnew, alpha=.6, color=cols[i], lw=3)
    scatter(embed_ring, h_t_ring[t,:] .|> φnew, alpha=.3, color=cols[i], s=5)
end
tight_layout()
yticks([0,R], [L"0",L"R"])
xticks([0,2*pi], [L"0",L"2\pi"])

angles=[6π/10, π].+π/5
# angles=[π,3π/2]
inds = [findmin(abs.(embed_ring .- angle))[2] for angle in angles]
vlines(angles, ymin=0, ymax=R, ls=":", color="k", alpha=.5)
for (i,t) in enumerate(t_show)
    r_ring = h_t_ring[t,inds] .|> φnew
    scatter(embed_ring[inds], h_t_ring[t,inds] .|> φnew, alpha=1, color=cols[i], s=60, edgecolor="k")
end


## ######################################
########## Plot field on the plane
########################################

timepoints_2 = ([0,50,100].+1070) ./dt .|> round.|>Int

embed_gaus = rotate(-π/2)(counts_gaus[:,1:2]')'

fig2b = PyPlot.figure(figsize=(10,10/3))
for (i,t) in enumerate(timepoints_2)
    subplot(1,length(timepoints_2),i)
    r_gaus = h_t_gaus[t,:] .|> φnew
    tripcolor(embed_gaus[:,1],embed_gaus[:,2],r_gaus, cmap="viridis", alpha=1, lw=0, ec="none")
    if i==1
        scatter(embed_gaus[:,1],embed_gaus[:,2], label="2D pop.", alpha=1/2, c="C1", s=1) ; 
    end
    # ticks=0:0.5: R
    xticks([]) ; yticks([]) 
    box(false)
end
# colorbar(label=L"r(t, z)")
tight_layout()



## #################### 
###### Ordered scatter plots

tshow = 700#times[end]/15
ind_tshow = tshow/dt |> round |> Int
sp = 20

inds = sortperm( atan.(embed_gaus[:,2],embed_gaus[:,1]) )
inds_show = inds[1:sp:N]

fig3=figure(figsize=(5,2.1))
raster_plot(spikes_gaus,ind_tshow, inds_show; sparsity=1, color="k")
