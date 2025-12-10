

########### Prepare external inputs.
##################### 
T_trial = 500

T_f = T_trial*N_trials #ms
times=dt:dt:T_f

times_trial = dt:dt:T_trial
cont_trial = (100 .< times_trial .< 300)
sense_trial = (100 .< times_trial .< 200 )

conts = [(-1)^n for n in 1:N_trials] 
s1 = [(-1)^fld(n+1,4) for n in 1:N_trials]
s2 = [(-1)^fld(n+1,2) for n in 1:N_trials]

context = cat( [conts[n]*cont_trial for n in 1:N_trials]..., dims=1)
sense1 = cat( [s1[n]*sense_trial for n in 1:N_trials]..., dims=1)
sense2 = cat( [s2[n]*sense_trial for n in 1:N_trials]..., dims=1)

context_strength = 1.5
input_strength =  1/10 

Icontext = context_strength *  1/τ * ( context ) ### (+) -> follow I1, (-) -> follow I2
I1 = input_strength * 1/τ .* (  sense1 .+ randn(length(times))/5 )
I2 = input_strength * 1/τ .* (  sense2 .+ randn(length(times))/5 )

;


## input to hidden structure : O.U. process
hidden_deriv = randn(length(times)) * dt * 1/τ * 1/50
input_hidden = zeros(length(times))
for i in 2:length(times); input_hidden[i] = input_hidden[i-1]*(1-dt/(10*τ)) + hidden_deriv[i]; end


inputs = [I1 I2 Icontext input_hidden] 