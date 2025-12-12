"""
Limit-cycle trajectories for two different models. See Fig. 3C-D.
"""


using PyPlot
using ProgressBars
pygui(true)

include("../src.jl")


using Random
Random.seed!(21)

# %%

δ = pi/10

τ = 10
R = 10*1/τ
J = 1
print("τJR = ", τ*J*R)

# step(x) = R*(x > 0) # transfer function

N = 2000

## ring:
J_ring = J
# M_ring = 1/2*[[1, 0] [0, 1]] |> inv
f_ring=(cos,sin)
g_ring=(z->J_ring*cos(z+δ),z->J_ring*sin(z+δ))
Z_ring = 2*pi*rand(N) |> sort!

r0_ring = 2.9348

## Gauss: σ and J_gaus match derivative at 0 and radius !!
σ =  0.6 #1/2*sqrt(π/2)
J_gaus = 1.39 * J_ring #J / (2σ^2) #sqrt(2/π) * J_ring # r0_ring /(τ*R) * sqrt(2*π)
# M_gaus = [[1, 0] [0, 1]] |> inv
f_gaus = (z->z[1],z->z[2])
g_gaus = (z->J_gaus*(rotate(δ)(z))[1] , z->J_gaus*(rotate(δ)(z))[2])
Z_gaus = [σ*randn(2) for n in 1:N]

# r0_gaus = τ * J_gaus * R / sqrt(2*pi) * cos(δ)
r0_gaus = r0_ring# 2.98 * cos(δ) 


# unit_κ = r0_ring


##########
# nodyn(x,y) = 0
r(x,y) = sqrt(x^2+y^2)
radial_F(r) = J_gaus*R * r*σ^2/sqrt(π)/sqrt(1+2*σ^2*r^2)
dkappa(κ) = -κ/τ + radial_F(r(κ[1],κ[2]))*rotate(δ)(κ)/r(κ[1],κ[2])
dx(x,y) = dkappa([x,y])[1]
dy(x,y) = dkappa([x,y])[2]

## ###################
using SpecialFunctions
φ(x) = R*(erf(x)+1)/2
# φ(x) = R*(x>0)

dt = 1e-1 # ms
T_f = 300 #ms
times=0:dt:T_f

conn_dis = false

κ0s = [[1.2 -1.3], [-0.1 0.1]]
κ_t_ring, κ_th_t_ring = run_from_init_conds(r0_ring*κ0s;
    spiking=true,φ=φ, times=times,  dx=dx, dy=dy, return_spikes=false, 
    f=f_ring, g=g_ring,  Z=Z_ring, conn_disorder=conn_dis)
κ_t_gaus, κ_th_t_gaus = run_from_init_conds(r0_gaus*κ0s;
    spiking=true,φ=φ, times=times,  dx=dx, dy=dy, return_spikes=false, 
    f=f_gaus, g=g_gaus,Z=Z_gaus, conn_disorder=conn_dis)

## ###########################
disp = 1.4
rax = -disp:0.01:disp
ray = -disp:0.01:disp


PyPlot.figure(figsize=(4,4))
plot_flow(rax,ray, dx,dy, unit_κ=r0_gaus)



pls = plot_traj.(κ_t_ring, unit_κ = r0_ring, marker="o",lw=0, alpha=1, #label=L"Low-rank SNN ($d=1$)", 
                color="C0", init_state=false)

pls2d = plot_traj.(κ_t_gaus, unit_κ = r0_gaus, marker=".",lw=0, alpha=.5, #label=L"Low-rank SNN ($d=2$)", 
                color="C1", init_state=true)

# pl = plot_traj.(κ_th_t_ring, unit_κ = r0_ring, color="C3", lw=2, init_state=true)
# pl[end][1].set_label("Neural field")

### LABELS
pls[end][1].set_label(L"Low-rank SNN ($d=1$)")# (N = $N)")
pls2d[end][1].set_label(L"Low-rank SNN ($d=2$)")# (N = $N)")
# pls2[end][1].set_label("Neural field")


phis = 0:0.1:(2*pi+0.1)
PyPlot.plot(cos.(phis),sin.(phis), "--", color="k", lw=2, label="Limit cycle")

# PyPlot.xlabel(L"\kappa_1 / r_0")
# PyPlot.ylabel(L"\kappa_2 / r_0")

PyPlot.legend(loc=1)

xticks(-1:1)
yticks(-1:1)

# PyPlot.subplots_adjust(left=0.2)

tight_layout()


## ####
figure(figsize=(2.5,2.5),frameon=false)
plot_flow(rax,ray, dx,dy, unit_κ=r0_gaus)
PyPlot.plot(cos.(phis),sin.(phis), "--", color="k", lw=2)
xticks([]); yticks([])
tight_layout()


## ##################
C = τ*R*cos(δ)

using SpecialFunctions
Phi_ring(x) = C*J*1/(2sqrt(π)) * x * exp(-x^2/2) * (besseli(1,x^2/2) + besseli(0,x^2/2))
Phi_gaus(x, σsq) = C*J_gaus*σsq * x / sqrt(π * (1+2x^2*σsq))

xs = 0:0.01:5

figure(figsize=(3,2))
plot(xs, Phi_ring.(xs), label=L"\Phi_{\rm ring}", lw=2, color="C0")
plot(xs, Phi_gaus.(xs, σ^2), label=L"\Phi_{\rm plane}", lw=2, color="C1")
plot(xs,xs, "--", color="k")
xlabel(L"\varrho")
ylabel(L"$\Phi(\varrho)$ [$\tau R$]")
legend()
# ylim(0,4)
tight_layout()
