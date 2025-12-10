
struct Model
    name::String
    generate_z::Function
    f::Vector{Function}
    g::Vector{Function}
    J::Float64
    unit_κ::Float64
    K::Float64
    
    function Model(name::String, generate_z::Function, f1::Function, f2::Function; J::Float64, unit_κ::Float64, K::Float64 = 0.)
        fI = z -> abs(f1(z)) - abs(f2(z))
        f_hid = z -> z[end]
        f = [f1, f2, fI, f_hid]
        mult_by(a) = (f -> (z -> a*f(z)))
        g = [mult_by(J)(f1), mult_by(J)(f2), (z -> 0.), mult_by(K)(f_hid)]
    
        new(name, generate_z, f, g, J, unit_κ, K)
    end 
end

struct Network
    name::String
    Model::Model
    Z::Vector{Vector{Float64}}
    
    function Network(model::Model, Z::Vector{Vector{Float64}}, ξs)
        N = length(Z)
        Z = [[Z[n]..., ξs[n]] for n in 1:N]
        new(model.name, model, Z)
    end
    
    function Network(model::Model, N::Int, ξs)
        Z = [model.generate_z() for n in 1:N]
        Network(model, Z, ξs)
    end
end    


######## HOW TO RUN A NETWORK
function run_network(network::Network, times, κ0, inputs, φ)

    h_t, κ_t, spk = run_ext_input(
        f=network.Model.f, g=network.Model.g, 
        Z=network.Z, κ0 = κ0, φ=φ, times=times, κI = inputs, 
        return_spikes=true
    )
    return h_t, κ_t, spk
end


############### STORED RESULTS

struct Result
    name::String
    Z::Vector{Vector{Float64}}

    h_t::Array{Float64,2}
    κ_t::Array{Float64,2}
    spikes::Array{Float64,2}

    compressed::Bool

    function Result(network::Network, h_t, κ_t, spikes; compressed=true)
        if compressed
            h_t, κ_t, spikes = (h_t, κ_t, spikes) .|> compress_array(10)
        end
        new(network.name, network.Z, h_t, κ_t, spikes, compressed)
    end

end