
""" 
Models:
- 'ring': 1D ring population, z in  [0,2π[
- 'MS': mixed selectivity, z = (z1,z2) in  [-1,1] x [-1,1]
- 'PS': pure selectivity, z = (z1,z2) in  [-1,1]x{0} U {0}x[-1,1]
- 'clust': clustered population, z in  {(±1,0), (0,±1)}
- 'hid': hidden feature, z = (z,ξ) in  [0,2π[ x [-1,1]
"""

# δ=1 
using SpecialFunctions
φ(x) = R*(erf((x-1))+1)/2 # transfer function


include("defs.jl")
##############################
# Models
ring_model = Model(
    "ring", 
    () -> rand(1)*2π,
    z->cos(z[1]), z->sin(z[1]), 
    J = 0.65, unit_κ = 1.31
)


MS_model = Model(
    "MS", 
    () -> 2*rand(2).-1,
    z -> z[1], z -> z[2], 
    J = 1., unit_κ = 1.38
)


unit_unif = () -> rand()*2-1
gen_z_PS = () -> (rand() > 0.5 ? [unit_unif(),0] : [0,unit_unif()])

PS_model = Model(
    "PS", 
    gen_z_PS,
    z -> z[1], z -> z[2], 
    J = 1.5, unit_κ = 1.33
)

clust_model = Model(
    "clust", 
    () -> sign.(gen_z_PS()),
    z -> z[1], z -> z[2], 
    J = 0.6, unit_κ = 1.30
)


####### model with hidden feature (K non zero)

hid_model = Model(
    "hid",
    ring_model.generate_z,
    ring_model.f[1], ring_model.f[2],
    J = ring_model.J, unit_κ = ring_model.unit_κ,
    K = 0.39 ####### to be determined more precisely!! (0.4 originally)
)



############ Models to use:

models = [
    clust_model,
    ring_model, 
    MS_model,   
    PS_model, 
    hid_model
]


models = Dict([model.name => model for model in models])

println("Models: ", [name for (name,_) in models])