module thermal_modeling

using Flux
using DiffEqFlux:FastChain, FastDense


mutable struct HeatTransferLayer{U,V,T}
    n_temps::Int
    n_targets::Int
    conductance_net::Dense{U,Matrix{V},Vector{T}}
    adj_mat::Matrix{Int8}
end

function HeatTransferLayer(n_input::Integer, n_temps::Integer, n_targets::Integer)
    # populate adjacency matrix
    adj_mat = zeros(Int8, n_temps, n_temps)
    k = 1
    for col_j in 1:n_temps
        for row_i in col_j + 1:n_temps
            adj_mat[row_i, col_j] = k
            k += 1
        end
    end
    adj_mat = adj_mat + adj_mat'
    n_conds = Int(0.5 * n_temps * (n_temps - 1))
    HeatTransferLayer(n_temps, n_targets,
                      Dense(n_input + n_targets, n_conds, σ),
                      adj_mat)
end

# overload struct to make it callable
function (m::HeatTransferLayer)(all_input)
    n_temps = m.n_temps
    prev_out = @view all_input[1:m.n_targets, :]
    temps = @view all_input[1:n_temps, :]
    
    conductances = m.conductance_net(all_input)
    
    # subtract, scale, and sum
    tmp = hcat([sum(temps[j, :] .- prev_out[i, :] .* conductances[m.adj_mat[i, j], :] 
                for j in 1:n_temps if j != i) 
                    for i in 1:m.n_targets]...)'
    # mutating arrays not allowed in zygote
    """tmp = zeros(eltype(prev_out), size(prev_out))
    for i in 1:m.n_targets
        for j in 1:n_temps
            if j != i
                @. tmp[i, :] += (temps[j, :] - prev_out[i, :]) * conductances[m.adj_mat[i, j], :]
            end
        end
    end"""


    return tmp
end

# specify what is trainable 
Flux.@functor HeatTransferLayer (conductance_net,)

mutable struct TNNCell{U <: Chain,V <: Real,S}
    sample_time::V
    ploss_net::U
    heat_net::HeatTransferLayer
    caps::Vector{V}
    prll::Parallel  # will be defined in inner constructor (no outer definition)
    state0::S
    function TNNCell(sample_time::V, ploss_net::U, heat_net::HeatTransferLayer, caps::Vector{V}, init_hidden::S) where {U <: Chain,V <: Real,S}
        new{U,V,S}(sample_time, ploss_net, heat_net, caps, Parallel(+, ploss_net, heat_net), init_hidden)
    end
end


function TNNCell(n_input::U, n_temps::U, n_targets::U, init_hidden::S) where {U <: Integer,S}
    ploss_net = Chain(Dense(n_input + n_targets, 8, σ),
                      Dense(8, n_targets, σ))
    heat_transfer = HeatTransferLayer(n_input, n_temps, n_targets)
    caps = 0.5f0 .* randn(Float32, n_targets) .- 3f0  # Gaussian mean=-3 std=0.5
    TNNCell(Float32(0.5), ploss_net, heat_transfer, caps, init_hidden)
end

function (m::TNNCell)(prev_̂y, x)
    x_non_temps, x_temps = x
    xx = vcat(prev_̂y, x_temps, x_non_temps)
    rh_ode = m.prll(xx)
    y = prev_̂y .+ m.sample_time .* 10f0.^m.caps .* rh_ode
    return y, prev_̂y
end

# specify what is trainable 
Flux.@functor TNNCell (ploss_net, heat_net, caps)

end # module
	