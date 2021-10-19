using Flux, DataFrames, CSV, ProgressMeter, DiffEqFlux, LinearAlgebra, DifferentialEquations, Plots

data =
    CSV.File(
        joinpath(pwd(), "..", "..", "..", "data", "measures.csv");
        typemap = Dict(Float64 => Float32),
    ) |> DataFrame
@. data[!, :i_norm] = sqrt(data.i_d^2 + data.i_q^2);

gdf = groupby(data, :profile_id)
# just work on profile 60 for now (dummy)
p6 = get(gdf, (profile_id = 60,), nothing);

target_cols = ["pm", "stator_tooth", "stator_winding", "stator_yoke"]
temp_cols = [target_cols; ["ambient", "coolant"]]
input_cols = [c for c in names(p6) if c ∉ [temp_cols; ["profile_id", "torque"]]]

n_temps = size(temp_cols, 1)
n_conds = Int(0.5 * n_temps * (n_temps - 1))

# normalize
p6[:, temp_cols] ./= 100
p6[:, input_cols] ./= Array(combine(p6[:, input_cols], input_cols .=> maximum))

# Initial condition
u0 = Array(p6[1, target_cols]) 

# Simulation interval and intermediary points
tspan = (0.0, nrow(p6) - 1)
tsteps = 0:nrow(p6)-1


conductance_net = FastChain((x, p) -> x, StaticDense(length(input_cols), n_conds, σ))
conductance_net_psize = DiffEqFlux.paramlength(conductance_net)

ploss_net = FastChain(
    (x, p) -> x,
    StaticDense(length(input_cols), 8, tanh),
    StaticDense(8, length(target_cols)),
)
ploss_net_psize = DiffEqFlux.paramlength(ploss_net)

sample_time = 0.5

function tnn!(dx, x, p, t)
    ex_idx = clamp(Int(round(t)), 1, nrow(p6))
    #conductance_net_params, loss_net_params, caps_params = p
    conductance_net_params = p[begin:conductance_net_psize]
    loss_net_params = p[conductance_net_psize+1:conductance_net_psize+ploss_net_psize]
    caps_params = p[conductance_net_psize+ploss_net_psize+1:end]
    conductances = conductance_net(Array(p6[ex_idx, input_cols]), conductance_net_params)
    loss_p = ploss_net(Array(p6[ex_idx, input_cols]), loss_net_params)
    current_temps = [x; Array(p6[ex_idx, ["ambient", "coolant"]])]


    #temp_diffs = [
    #    sum(current_temps[j] - x[i] * conductances[1] #adj_mat[i, j]]
    #        for j in 1:n_temps if j != i) for i = 1:size(target_cols, 1)
    #]
    
    temp_diffs = zeros(eltype(caps_params), size(caps_params))
    @inbounds @simd for i in 1:size(target_cols, 1)
        @inbounds @simd for j in 1:n_temps
            if j != i
                # nightmarish workaround to avoid indexing with i, j
                #  since that is apparently not differentiable
                if j > i
                    cond_idx = (i-1)*(n_temps-i) + j - 1 + i ÷ 2 + 2*(i ÷ 3)
                else
                    cond_idx = (j-1)*(n_temps-j) + i - 1 + j÷2 + 2*(j ÷ 3)
                end
                @. temp_diffs += (current_temps[j] - x[i]) * conductances[cond_idx]
            end
        end
    end

    dx[:] .= 10 .^ caps_params .* sample_time .* (loss_p .+ temp_diffs)
end


prob_neuralode = ODEProblem(
    tnn!,
    u0,
    tspan,
    [
        initial_params(conductance_net),
        initial_params(ploss_net),
        (-1 .- 2 .* rand(Float32, size(target_cols, 1))),
    ],
)

function predict_neuralode(p)
    tmp_prob = remake(prob_neuralode; p = p)
    Array(solve(
        tmp_prob,
        Tsit5(), #sensealg=ReverseDiffAdjoint(), 
        saveat = tsteps,
    ))
end

scaled_targets = Array(p6[!, target_cols]) 

function loss_neuralode(p)
    pred = predict_neuralode(p)
    #println(size(scaled_targets), size(pred))
    loss = sum(abs2, scaled_targets .- pred') # Just sum of squared error
    return loss, pred
end

plot_cb = function (p, l, pred; doplot = true)
    display(l)
    # plot current prediction against data
    plt = plot(tsteps, scaled_targets, label = "data")
    plot!(plt, tsteps, pred', label = "prediction")
    annotate!(0.5, 2, text(l, :left, 10))
    if doplot
        display(plt)
    end

    return false
end

println("Start training ..")
result_neuralode =
    DiffEqFlux.sciml_train(loss_neuralode, reduce(vcat, prob_neuralode.p), cb = plot_cb)


# Setup the ODE problem, then solve
#prob = ODEProblem(lptn_1d!, u0, tspan, p)
#sol = solve(prob)
