using ProgressBars
using ProgressMeter
using LaTeXStrings
using SparseArrays


# global M, f1, f2, g1, g2, N, Z


rotate(φ) = v -> [[cos(φ), sin(φ)] [-sin(φ), cos(φ)]]*v 
plot_point(v,args...;marker="o", s=80, kwargs...) = PyPlot.scatter(v...,args...; marker=marker, s=s, kwargs...)



compress_array(factor=10) = x -> selectdim(x,1,1:10:size(x,1)) .|> Float16
        
to_single_vec(κ_t) = reduce(vcat,κ_t) 

function interrupt()
    print("\n------------------- ** Interrupted ** ------------------- \n\n")
    throw(InterruptException)
end


function meshgrid(x, y)
    X = [i for i in x, j in 1:length(y)]
    Y = [j for i in 1:length(x), j in y]
    return X, Y
end
heatmap(z,rax,ray;kwargs...)= PyPlot.imshow(z',extent=[rax[[1, end]]...,ray[[1, end]]...],origin="lower"; kwargs...)


function low_pass_filter(x; α=1e-2) # lowpass filter in direcion of rows of x
    y = zeros(size(x))
    y[:,1] = x[:,1]
    for i in 2:size(x,2)
        y[:,i] = (1-α)*y[:,i-1] + α*x[:,i]
    end
    return y
end

# using DSP.Filters
# function LP_filter(x; frequ=1-1e-2)
#     return x |> Lowpass(frequ)
# end



function Z_score(spikes; α=1e-2, return_LPspikes=false)
    LPspikes = low_pass_filter(spikes',α=α)
    z_scores = (LPspikes .- mean(LPspikes, dims=2)) ./ std(LPspikes, dims=2) |> transpose
    # z_scores = LPspikes |> transpose
    if return_LPspikes
        return z_scores, LPspikes
    end
    return z_scores
end



#########

##########
using Distributions


global dt, times
function run_(;
    κ0=κ0,spiking=false, φ=x->(x>0), times=times, dt=dt, 
    f=(f1,f2), g=(g1,g2), M=nothing, Z=Z, dx=dx, dy=dy, return_spikes=false,
    Law_χ=Normal(0,1), conn_disorder = false )

    (f1,f2) = (f[1],f[2]) ; (g1,g2) = (g[1],g[2])

    F = [f1.(Z)  f2.(Z)]'
    G = [g1.(Z)  g2.(Z)]'


    if isnothing(M)
        M = F*F'/N |> inv
    end

    if conn_disorder ; χ=rand(Law_χ,(N,N)) ; end

    κ_from_h(h) = (h' * F')*M' / N
    h_from_κ(κ) = F' * κ'[:,1]

    # vec_f(z) = [f1(z), f2(z)]   

    function push_state(ind_,h_,κ_, κ_th_)
        κ_t[ind_,:] = κ_
        h_t[ind_,:] = h_
        κ_th_t[ind_,:] = κ_th_
        return
    end

    ################## init
    h_t = zeros(length(times),N)
    κ_t = zeros(length(times),size(F)[1])
    κ_th_t = zeros(length(times),size(F)[1])
    if return_spikes; spikes = zeros(Bool, length(times),N); end
    
    h = h_from_κ(κ0) #
    κ = κ_from_h(h)
    κ_th = κ0'
    
    push_state(1,h,κ, κ_th)
    
    ###################


    # for ind in ProgressBars.ProgressBar( 2:length(times) )
    @showprogress for ind in 2:length(times)
        if spiking
            output = (rand(N) .< φ.(h)*dt)  / dt
        else
            output = φ.(h)
        end
        κrec = G*output / N
        h = (1-dt/τ)*h + F'*κrec * dt 
        if conn_disorder ; h += χ * output /N * dt ; end
        κ = κ_from_h(h)

        κ_th = κ_th + dt*[dx(κ_th[1],κ_th[2]),dy(κ_th[1],κ_th[2])]

        push_state(ind,h,κ, κ_th)
        if return_spikes ; spikes[ind,:] = output.!=0 ; end
    end

    if !return_spikes
        return h_t,κ_t, κ_th_t
    else 
        return h_t,κ_t, κ_th_t, spikes
    end
end

function run_from_init_conds( κ0s; run_function = run_, return_h=false, return_spikes=false, kwargs...)
    h_t_s = []
    κ_t_s = []
    κ_th_t_s = []
    spk_s = []

    for κ0 in κ0s
        print(κ0)
        output = run_function(;κ0=κ0, return_spikes=return_spikes, kwargs...)
        h_t, κ_t, κ_th_t = output[1], output[2], output[3]
        push!(h_t_s,h_t)
        push!(κ_t_s,κ_t)
        push!(κ_th_t_s,κ_th_t)
        if return_spikes
            push!(spk_s, output[4])
        end
    end
    if return_h
        return κ_t_s , κ_th_t_s, h_t_s
    end
    if return_spikes 
        return κ_t_s , κ_th_t_s, h_t_s, spk_s
    end
    return κ_t_s , κ_th_t_s
end

function plot_traj(κ_t;unit_κ=unit_κ, step_=10, init_state=false, final_state = false, kwargs...) 
    pl=PyPlot.plot(κ_t[1:step_:end,1]/unit_κ,κ_t[1:step_:end,2]/unit_κ; kwargs...)
    if init_state
        PyPlot.plot(κ_t[1,1]/unit_κ,κ_t[1,2]/unit_κ; color = "black", marker="*")
    end
    if final_state
        PyPlot.plot(κ_t[end,1]/unit_κ,κ_t[end,2]/unit_κ; color = "red", marker=".")
    end
    return pl
end

# using Colors
function plot_flow(rax,ray, dx,dy; unit_κ=unit_κ, alpha=1, plot_speed = true, density=1)
    xs,ys=meshgrid(rax,ray)
    fx = dx.((xs,ys).*unit_κ...)
    fy = dy.((xs,ys).*unit_κ...)
    speeds = sqrt.(fx.^2+fy.^2)
    # gray = alpha*[0.5,0.5,0.5] 
    # gray = RGB(gray...)
    str=streamplot(xs',ys',fx',fy',  linewidth=.5, color=:gray, density=density, zorder=0) 
    if plot_speed
        heatmap(speeds,rax,ray,alpha=0.6*alpha,cmap="GnBu", zorder=-1)
    end
end


function plot_trajs_models(κ_t_s, κ_th_t_s, rax,ray,dx,dy; newfig=true)
    if newfig; PyPlot.figure(figsize=(5,5)) end

    plot_flow(rax,ray, dx,dy)

    pls = plot_traj.(κ_t_s,marker="o", markersize=4,lw=.1, color="black")
    ### analytical traj.
    pls2 = plot_traj.(κ_th_t_s, color="C3", lw=2)

    return pls,pls2
end

function embed_spikes(spikes; time_points = time_points, nsteps=100)
    Dim = length(time_points)
    emb = zeros((Dim,N))
    for (i,p) in enumerate(time_points)
        emb[i,:] = sum(spikes[p:p+nsteps,:], dims=1)/nsteps/dt 
    end
    return emb
end

function embed_spikes_intervals(spikes, intervals)
    Dim = length(intervals)
    emb = zeros((Dim,N))
    for (i,interval) in enumerate(intervals)
        emb[i,:] = sum(spikes[interval[1]:interval[2],:], dims=1)/(interval[2]-interval[1])/dt 
    end
    return emb
end

function scat_embed(h_t, time_points; φ=φ, kwargs...)
    embed(h_t, time_points=time_points) = h_t[time_points,:] .|> φ
    scatter3D(embed(h_t)[1,:],embed(h_t)[2,:],embed(h_t)[3,:]; kwargs...)
end


function to_vec_tuple(m)
    vec_tuple = [Tuple(row) for row in eachrow(m)]
    return vec_tuple
end

function points(PersDiag; inf=nothing)
    if inf === nothing; inf=PersDiag.threshold; end
    m = PersDiag .|> (diag -> [diag.birth, min(diag.death,inf)])
    return reduce(hcat,m)
end

function plot_persistence_diagram(PersDiags; s=20, kwargs...)
    for (i,rip) in enumerate(PersDiags)
        if length(rip) != 0
            scatter(points(rip)[1,:],points(rip)[2,:], s=s, label="H_$(i-1)", color="C$(i-1)", kwargs...)
        end
    end
    plot([0,PersDiags[1].threshold], PersDiags[1].threshold*[1,1], "k:", label=L"\infty", lw=1)
    plot(PersDiags[1].threshold*[0,1],PersDiags[1].threshold*[0,1], "k", lw=1)
end

function plot_barcode(PersDiags; lw=2,  kwargs...)
    y=0
    pls=[]
    for (k,rip) in enumerate(PersDiags)
        pl=nothing
        barcodes = points(rip)
        for i in 1:size(barcodes,2)
            pl = plot(barcodes[:,i], [y, y],"-", lw=lw, c="C$(k-1)", kwargs... )
            y += 1
        end
        append!(pls,pl)
    end
    return pls
end

function plot_single_barcode(pers; start=0, lw=2,  kwargs...)
    bars = points(pers)
    y=start
    for i in 1:size(bars,2)
        pl = plot(bars[:,i], [y, y],"-", lw=lw; kwargs... )
        y += 1
    end
    return y
end



function raster_plot(spikes,ind_tshow, inds_show; sparsity=1, kwargs...)
    Nshow = length(inds_show)
    for (y,i) in enumerate(inds_show)
        vlines(times[1:sparsity:ind_tshow][spikes[1:sparsity:ind_tshow,i] .!= 0], ymin=y, ymax=y+1; kwargs...)
    end
    xlim((0,tshow))
    xlabel("time [ms]")
    yticks([Nshow,1],[L"N","1"])
    # ax2.set_ylabel("neuron index")
    ylim(((1,Nshow)))
    tight_layout()
end


##################
# function run_nodisorder_ext_input(;κ0=κ0,spiking=true, φ=x->(x>0), times=times, dt=dt, f=(f1,f2), g=(g1,g2), M , Z=Z, 
#     fI=(z->0*f1(z)), gI = (z->0*f1(z)), κI = zeros(length(times),3))
    
#     """
#     DEPRECATED
#     """
    
#     (f1,f2) = (f[1],f[2]) ; (g1,g2) = (g[1],g[2])

#     F = [f1.(Z)  f2.(Z) fI.(Z)]'
#     G = [g1.(Z)  g2.(Z) gI.(Z)]'

#     # χ=rand(Law_χ,(N,N))

#     κ_from_h(h) = (h' * F')*M' / N
#     h_from_κ(κ) = F' * κ'[:,1]

#     # vec_f(z) = [f1(z), f2(z)]   

#     function push_state(ind_,h_,κ_)#, κ_th_)
#         κ_t[ind_,:] = κ_
#         h_t[ind_,:] = h_
#         #κ_th_t[ind_,:] = κ_th_
#         return
#     end

#     ################## init
#     h_t = zeros(length(times),N)
#     κ_t = zeros(length(times),size(F)[1])
#     #κ_th_t = zeros(length(times),size(F)[1])
#     #if return_spikes; spikes = zeros(length(times),N); end
    
#     h = h_from_κ(κ0) #
#     κ = κ_from_h(h)
#     #κ_th = κ0'
    
#     push_state(1,h,κ)#, κ_th)
    
#     ###################


#     for ind in ProgressBars.ProgressBar( 2:length(times) )
#         if spiking
#             # output = (rand(N).<R*dt) .* (h.>0)  / dt
#             output = (rand(N) .< φ.(h)*dt)  / dt
#         else
#             output = φ.(h)
#         end
#         κrec = G*output / N
#         h = (1-dt/τ)*h + F'* (κrec + κI[ind,:]) * dt  # + Iext[ind] * fI.(Z) .* ξ * dt# +  χ * output /N * dt
#         κ = κ_from_h(h)

#         #κ_th = κ_th + dt*[dx(κ_th[1],κ_th[2]),dy(κ_th[1],κ_th[2])]

#         push_state(ind,h,κ)#, κ_th)
#         #if return_spikes ; spikes[ind,:] = output.!=0 ; end
#     end

#     return h_t,κ_t, nothing
# end

###################################################
###################################################
###################################################
mat_from_func(fun_vec,Z) = [fi.(Z) for fi in fun_vec] |> (x->reduce(hcat,x)) |> transpose

"""
Runs the model with low-dimensional external input κI.
Returns membrane potentials h_t, latent variables κ_t, and spikes if return_spikes=true.
"""

function run_ext_input(; f, g, Z, κ0, φ, times, 
    spiking=true, dt=dt, κI = nothing, M = nothing,
    conn_disorder = false, Law_χ=Normal(0,1),
    return_spikes=false, rate_heterogeneity=nothing)

    local F
    F = mat_from_func(f,Z)
    G = mat_from_func(g,Z)

    if isnothing(M)
        M = F*F'/N |> inv
    end

    if isnothing(κI)
        κI = zeros(length(times),size(F)[1])
    end

    if conn_disorder
        χ=rand(Law_χ,(N,N))
    end

    κ_from_h(h) = (h' * F')*M' / N
    h_from_κ(κ) = F' * κ'[:,1]

    ################## init
    h_t = zeros(length(times),N)
    κ_t = zeros(length(times),size(F)[1])
    if return_spikes; spikes = zeros(Bool , length(times),N); end


    function push_state(ind_,h_,κ_)#, κ_th_)
        κ_t[ind_,:] = κ_
        h_t[ind_,:] = h_
        #κ_th_t[ind_,:] = κ_th_
        return
    end
    
    h = h_from_κ(κ0)
    κ = κ_from_h(h)
    
    push_state(1,h,κ)#, κ_th)
    

    rates(h) = if isnothing(rate_heterogeneity) φ.(h) else φ.(h) .* rate_heterogeneity end
    # for ind in ProgressBars.ProgressBar( 2:length(times) )
    @showprogress for ind in 2:length(times)
        if spiking
            output = (rand(N) .< rates(h)*dt)  / dt
        else
            output = rates(h)
        end
        κrec = G*output / N
        h = (1-dt/τ)*h + F'* (κrec + κI[ind,:]) * dt 
        if conn_disorder
            h += χ * output /N * dt
        end

        κ = κ_from_h(h)

        push_state(ind,h,κ) 
        if return_spikes ; spikes[ind,:] = output.!=0 ; end
    end

    if return_spikes
        return h_t,κ_t, sparse(spikes)
    else
        return h_t,κ_t, nothing
    end
end



###################################################
###################################################
## NOT USED
function run_full_ext_input(; f, g, Z, κ0, φ, times, ext_input, 
    spiking=true, dt=dt,
    conn_disorder = false, Law_χ=Normal(0,1), return_spikes=false)

    local F
    F = mat_from_func(f,Z)
    G = mat_from_func(g,Z)

    if conn_disorder
        χ=rand(Law_χ,(N,N))
    end

    M = F*F'/N |> inv
    
    κ_from_h(h) = (h' * F')*M' / N
    h_from_κ(κ) = F' * κ'[:,1]

    ################## init
    h_t = zeros(length(times),N)
    κ_t = zeros(length(times),size(F)[1])
    if return_spikes; spikes = zeros(Bool, length(times),N); end


    function push_state(ind_,h_,κ_)#, κ_th_)
        κ_t[ind_,:] = κ_
        h_t[ind_,:] = h_
        #κ_th_t[ind_,:] = κ_th_
        return
    end
    
    h = h_from_κ(κ0)
    κ = κ_from_h(h)
    
    push_state(1,h,κ)#, κ_th)
    


    rates(h) = φ.(h)

    for ind in ProgressBars.ProgressBar( 2:length(times) )
        if spiking
            output = (rand(N) .< rates(h)*dt)  / dt
        else
            output = rates(h)
        end
        κrec = G*output / N
        h = (1-dt/τ)*h + F'* κrec * dt  + ext_input[ind,:] * dt
        if conn_disorder
            h += χ * output /N * dt
        end

        κ = κ_from_h(h)

        push_state(ind,h,κ) 
        if return_spikes ; spikes[ind,:] = output.!=0 ; end
    end

    if return_spikes
        return h_t,κ_t, spikes
    else
        return h_t,κ_t, nothing
    end
end


