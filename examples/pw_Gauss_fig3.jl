"""
Example: 2D dynamics with 1D circuit space. See Fig. 3A-B.
"""

using PyPlot
using ProgressBars
pygui(true)

using Random    
Random.seed!(4321)

J = .8
τ = 10
R = 10*1/τ
α = π/3
a = cos(α)  ## 
b = -sin(α) ## 

# transfer function
using SpecialFunctions
φ(x) = R * (1 + erf(x))/2



M = ([[1+a^2, a*b] [a*b, b^2]] /2) |> inv
# M=([[b^2, a*b] [a*b,1+a^2 ]]/2) |> inv
f1(z) = z*((z>0) + a*(z<0))
f2(z) = b*z*(z<0)
g1(z) = J*f1(z)
g2(z) = J*f2(z)

N = 2000
Z = randn(N) |> sort!
# Z = (Z.>0)*2 .- 1 ## binarise the pattern: works as well!

# r0 = τ*R/(pi)*J*cos(δ)
k = J * τ * R / sqrt(2*π)
unit_κ = k

include("../src.jl")


################## Latent Dynamics
# Fmat=[[1,0] [a, b]]'
# Finv = Fmat |> inv
# κ1 = k*Finv*[1, a]
# κ3 = k*Finv*[-a,-1]

# dkappa(κ) = 1/τ  * (-κ + ((Fmat*κ)[1]>0)*κ1 + ((Fmat*κ)[2]<0)*κ3)
# dx(x,y) = dkappa([x,y])[1]
# dy(x,y) = dkappa([x,y])[2]


### For erf transfer function

Φ(x) = (1+x/sqrt(x^2 + 1/2))/(2*sqrt(2π))
Amat = [[1,0] [a, b]]
λ_(κ) = Amat'*κ
dkappa(κ) = -κ/τ + J*R*Amat*[Φ(λ_(κ)[1]), -Φ(-λ_(κ)[2])]
dlambda(λ) = -λ/τ + J*R*Amat'*Amat*[Φ(λ[1]), -Φ(-λ[2])]

dx(x,y) = dkappa([x,y])[1]
dy(x,y) = dkappa([x,y])[2]

## Find fixed points
using NLsolve
saddle1 = nlsolve(dkappa, [0.,1.]*k).zero
saddle2 = nlsolve(dkappa, [1.,1/2]*k).zero
fp1 = nlsolve(dkappa, [-1/2,1.]*k).zero
fp2 = nlsolve(dkappa, [1.,-1/2]*k).zero
fp3 = nlsolve(dkappa, [1.,1.]*k).zero
 

# rax = -1.5:0.01:1.5
# ray = -1.5:0.01:1.5
# plot_flow(rax,ray,dx,dy)

##################### time evol.
dt = 1e-1 # ms
T_f = 300 #ms
times=0:dt:T_f

κ0s = k*[
    [1.2 -1], 
    [1.2 1.2] , 
    [-1 -.1] ,
    # [-.5+.1 -√(3)/2]/2 , 
    [-.5-.1 -√(3)/2]*0.6 , 
    ]

κ_t_s , κ_th_t_s, _, _ = run_from_init_conds(κ0s;spiking=true ,φ=φ, M=M, conn_disorder=false, return_spikes=true)


##################### Plot

sq = 1.3
rax = -sq:0.01:sq
ray = -sq:0.01:sq

PyPlot.figure(figsize=(4,4))
plot_flow(rax,ray, dx,dy)


pls = plot_traj.(κ_t_s,marker="o", markersize=4,lw=.1, color="C0", init_state=true, unit_κ=k)
### analytical traj.
# pls2 = plot_traj.(κ_th_t_s, color="C3", lw=2, init_state=false, alpha=.5)


### LABELS
pls[end][1].set_label("Low-rank SNN")# (N = $N)")
# pls2[end][1].set_label("Neural field")


fp_style = Dict(:s=>50, :zorder=>10, :color=>"g")
plot_point.((fp1,fp2,fp3)./k; fp_style...)
scatter([],[]; fp_style..., label="Stable FP")

saddle_style = Dict(:s=>50, :edgecolors=>"r", :facecolors=>"none", :zorder=>10)
plot_point.((saddle1, saddle2)./k; saddle_style...)
scatter([],[]; saddle_style..., label="Saddle point")

xticks([-1,0,1])
yticks([-1,0,1])


PyPlot.legend()

tight_layout()



# # ##############
figure(figsize=(2.5,2.5),frameon=false)
plot_flow(rax,ray, dx,dy)
plot_point.((fp1,fp2,fp3)./k; fp_style...)
plot_point.((saddle1, saddle2)./k; saddle_style...)
xticks([]); yticks([])
tight_layout()

# # xlim(-2,3) ; ylim(-1,4)
# PyPlot.box(false) ; PyPlot.xticks([]) ; PyPlot.yticks([])

######################
map = "winter"; limv = 1
figure(figsize=(4,2))
scatter(Z,0*Z, c=Z, 
    cmap=map,
    vmin=-limv, vmax=limv, alpha=exp.(-Z.^2/2)*0.1
)
    xticks([]); yticks([])
box(false)
# plot(Z,exp.(-Z.^2/2))
tight_layout()


########### SIMILARITY SPACE
## RUN FOR MANY INITIAL CONDITIONS
n_traj = 20
T_f = 300 #ms
times=0:dt:T_f

κ0s = [k/2*(rand(2) .- 1/2)' for _ in 1:n_traj] 
# κ0s = [randn(2)' * k for _ in 1:n_traj]

_ , _, h_t_s, spk_s = run_from_init_conds(κ0s;spiking=true ,φ=φ, M=M, conn_disorder=false, return_spikes=true)

spk_s_LP = [low_pass_filter(spk'; α=1e-2)' for spk in spk_s]

# end_h =  h_t_s .|> (h->h[end,:])# mean.(h_t_s, dims=1)[:,1] 
# mean_spk = mean.(spk_s_LP, dims=1)
# figure()
# for n in 1:n_traj
#     scatter(end_h[n].|>φ, mean_spk[n], alpha=.3, s=18)
# end

using MultivariateStats
all_spks =  cat([spk for spk in spk_s_LP]..., dims=1)
pca = fit(PCA, all_spks'; maxoutdim=3)
embs = loadings(pca)


figure(figsize=(4,2))
scatter(embs[:,1],embs[:,2], 
    # alpha=.82, s=18, edgecolor="white")
    alpha=.3, s=18, 
    c = -Z, cmap=map, vmin=-limv, vmax=limv,
)
gca().set_aspect("equal")
xticks([]); yticks([])
box(false)
tight_layout()

print("Cumulative variance explained by PCs: ",
cumsum(principalvars(pca))/var(pca)
)
