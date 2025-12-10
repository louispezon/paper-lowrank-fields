include("../src.jl")
using Random

SEED = 1234 ; 
# SEED = 2 ; 
Random.seed!(SEED) 

using Printf


SAVE_RESULTS = true
COMPRESS = true



############ Hidden feature
N = 3000
ξs = (rand(N).>1/2).*2 .- 1

########### other parameters
τ = 10 # ms
R = 10*1/τ # kHz

dt = 1e-1 # ms

N_trials = 32

########### define models & inputs
include("setup_input.jl")

include("defs.jl")

include("models.jl")


###############################################################
κ0 = 0. *[0,0, 0, 0]'




################ ########### RUN NETWORKS
println("Models: ", [name for (name,_) in models])

println("number of trials: ", N_trials)

results = Dict{String, Result}()

for (name,model) in models
    println("Running model: ", model.name)
    net = Network(model, N, ξs)

    @time h_t, κ_t, spikes = run_network(net, times, κ0, inputs, φ)

    results[model.name] = Result(net, h_t, κ_t, spikes; compressed=COMPRESS)
end


############################ SAVE RESULTS
compress_factor = 10
comp_times = compress_array(compress_factor)(times)

K = models["hid"].K

using JLD2
function save_results(filename, path="data/")
    # jldsave(path * filename, results, comp_times, inputs, N, SEED, φ)
    @save path * filename results comp_times times inputs N SEED ξs dt τ R N_trials K
end

N_models = length(models)
filename = @sprintf("CDM%imodels_N%d_%itrials_K%.2f_seed%i",N_models,N, N_trials,K, SEED)*(COMPRESS ? "_compressed" : "")*".jld2"

if SAVE_RESULTS
    save_results(filename)
    println("data saved in $filename")
end


