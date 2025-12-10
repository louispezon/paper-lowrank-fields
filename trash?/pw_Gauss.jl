using PyPlot
using ProgressBars
pygui(true)

using Random    
Random.seed!(4321)

J = 1
τ = 10
R = 10*1/τ
α = 1/2

# transfer function
# φ(x) = R*(x >= 0) 


a=1e-3 
# φ(x) = R*(tanh(x/a)+1)/2
φ(x) = R * (x>-a)#+R/2*(x==0)
# plot(-5:0.1:5,φ.(-5:0.1:5))

β = -sqrt(1-α^2)
M = ([[1+α^2, α*β] [α*β, β^2]] /2) |> inv
# M=([[β^2, α*β] [α*β,1+α^2 ]]/2) |> inv
f1(z) = z*((z>0) + α*(z<0))
f2(z) = β*z*(z<0)
g1(z) = J*f1(z)
g2(z) = J*f2(z)

N = 2000
Z = randn(N) |> sort!

# r0 = τ*R/(pi)*J*cos(δ)
k = J * τ * R / sqrt(2*π)
unit_κ = k

include("../src.jl")


################## Latent Dynamics
Fmat=[[1,0] [α, β]]'
Finv = Fmat |> inv
κ1 = k*Finv*[1, α]
κ3 = k*Finv*[-α,-1]

dkappa(κ) = 1/τ  * (-κ + ((Fmat*κ)[1]>0)*κ1 + ((Fmat*κ)[2]<0)*κ3)
dx(x,y) = dkappa([x,y])[1]
dy(x,y) = dkappa([x,y])[2]

# rax = -1.5:0.01:1.5
# ray = -1.5:0.01:1.5
# plot_flow(rax,ray,dx,dy)

##################### time evol.
dt = 1e-1 # ms
T_f = 300 #ms
times=0:dt:T_f

κ0s = k*[
    [-1 -.7] , 
    [1.2 -1], 
    [1.2 1.2] , 
    [-1 -.5]
    ]

κ_t_s , κ_th_t_s, h_t_s, spk_s = run_from_init_conds(κ0s;spiking=true ,φ=φ, M=M, conn_disorder=false, return_spikes=true)


##################### Plot

sq = 1.3
rax = -sq:0.01:sq
ray = -sq:0.01:sq

PyPlot.figure(figsize=(4,4))
plot_flow(rax,ray, dx,dy)

pls = plot_traj.(κ_t_s,marker="o", markersize=4,lw=.1, color="C0", init_state=true)
### analytical traj.
# pls2 = plot_traj.(κ_th_t_s, color="C3", lw=2, init_state=false, alpha=.5)


### LABELS
pls[end][1].set_label("Low-rank SNN")# (N = $N)")
# pls2[end][1].set_label("Neural field")



plot_point.((κ1,κ3)./k, color="g", zorder=10)
plot_point((κ1+κ3)/k, color="g", label= "Stable FP",zorder=10, s=50)
plot_point([0,0], edgecolors="r",facecolors="none", label="Saddle point", zorder=10) 

PyPlot.legend()

# PyPlot.xlabel(L"\kappa_1 \,/ \, k")
# PyPlot.ylabel(L"\kappa_2  \,/ \, k")  

# PyPlot.subplots_adjust(left=0.2)
tight_layout()



# # ##############
# figure(figsize=(3,3),frameon=false)
# scatter(f1.(Z),f2.(Z), alpha=.3, s=18)
# # xlim(-2,3) ; ylim(-1,4)
# PyPlot.box(false) ; PyPlot.xticks([]) ; PyPlot.yticks([])

######################

########### SIMILARITY SPACE
using MultivariateStats
all_spks =  cat([spk for spk in spk_s]..., dims=1)
pca = fit(PCA, all_spks'; maxoutdim=2)
embs = loadings(pca)

figure(figsize=(3,3))
scatter(embs[:,1],embs[:,2], alpha=.3, s=18)#,edgecolor="white")
gca().set_aspect("equal")
# xticks([]); yticks([])
# box(false)
tight_layout()