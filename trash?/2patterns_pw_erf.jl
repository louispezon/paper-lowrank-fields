using PyPlot
using ProgressBars
pygui(true)

using Random    
Random.seed!(4321)

J = 1
τ = 10
R = 10*1/τ
α = π/3
a = cos(α)  ## 
b = -sin(α) ## 

# transfer function
using SpecialFunctions
φ(x) = R * (1 + erf(x))/2
# φ(x) = R*(x>0)

## generate pure selectivity patterns
rot = [1 1 ; 1 -1]
gen_z() = rot*((rand(2).>1/2).* 2 .- 1)/2 .|> Int


## connectivity functions
f1_1d(z) = z*((z>0) + a*(z<0))
f2_1d(z) = b*z*(z<0)

f1(z) = 2*f1_1d(z[1]) + z[2]*(1-a)*(z[2]>0)
f2(z) = 2*f2_1d(z[1]) - z[2]*b*(z[2]>0)
g1(z) = J*f1(z)
g2(z) = J*f2(z)


N=500
Z = [gen_z() for i in 1:N] 


# r0 = τ*R/(pi)*J*cos(δ)
k = J * τ * R / sqrt(2*π)
unit_κ = k

include("../src.jl")


##################### time evol.
dt = 1e-1 # ms
T_f = 300 #ms
times=0:dt:T_f

κ0s = 2*k*[
randn(2)' for i in 1:20
]

κ_t_s , κ_th_t_s = run_from_init_conds(κ0s;
    run_function=run_ext_input,
    f=(f1,f2), g=(g1,g2), Z=Z, 
    times=times, dt=dt,
    spiking=true ,φ=φ
)

#####################
##################### plot
PyPlot.figure(figsize=(4,4))
pls = plot_traj.(κ_t_s,marker=".", markersize=4,lw=.1, color="C0", init_state=true, unit_κ=k, final_state=true)
### analytical traj.
# pls2 = plot_traj.(κ_th_t_s, color="C3", lw=2, init_stat

fv(z) = [f1(z), f2(z)]
plot_point.(fv.(Z),alpha=.2, marker=".", color="C1")