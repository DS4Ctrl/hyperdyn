using Flux, DataFrames, CSV, ProgressMeter, DiffEqFlux, LinearAlgebra, DifferentialEquations, Plots

u0 = Float32[2.0; 0.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))

dudt2 = FastChain((x, p) -> x.^3,
                  FastDense(2, 50, tanh),
                  FastDense(50, 2))
dudt!(u, p, t) = dudt2(u, p)
u0 = rand(2)
prob_neuralode = ODEProblem(dudt!, u0, tspan, initial_params(dudt2))
sol_node = solve(prob_neuralode, Tsit5(), saveat = tsteps)

function predict_neuralode(p)
  tmp_prob = remake(prob_neuralode, p = p)
  Array(solve(tmp_prob, Tsit5(), saveat = tsteps))
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
end

callback = function (p, l, pred; doplot = true)
  #display(l)
  # plot current prediction against data
  plt = scatter(tsteps, ode_data[1,:], label = "data")
  scatter!(plt, tsteps, pred[1,:], label = "prediction")
  annotate!(0.5, 2, text(l, :left, 10))
  if doplot
    display(plt)
  end
  return false
end

result_neuralode = DiffEqFlux.sciml_train(loss_neuralode, prob_neuralode.p, cb = callback)