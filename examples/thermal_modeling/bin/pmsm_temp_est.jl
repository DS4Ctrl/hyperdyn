using Flux, DataFrames, CSV, ProgressMeter
using Statistics:mean
using thermal_modeling:TNNCell


function main()
        
    data = CSV.File(joinpath(pwd(), "..", "data", "measures.csv")) |> DataFrame

    # FE
    @. data[!, :i_norm] = sqrt(data.i_d^2 + data.i_q^2)
    data[!, :fe1] = data.i_norm / maximum(data.i_norm) .* data.motor_speed / maximum(data.motor_speed);

    # normalization
    # todo

    test_set_pids = [60, 62, 74]
    target_cols = ["pm", "stator_tooth", "stator_winding", "stator_yoke"]
    gdf = groupby(data, :profile_id)

    p_sizes = combine(gdf, nrow)
    max_len_test = maximum(filter(:profile_id => n -> n in test_set_pids, p_sizes).nrow)
    max_len_train = maximum(filter(:profile_id => n -> n ∉ test_set_pids, p_sizes).nrow)

    n_test_profiles = length(test_set_pids)
    n_train_profiles = length(keys(gdf)) - n_test_profiles

    c_input_temps = ["ambient", "coolant"]
    c_temps = [target_cols..., c_input_temps...]
    c_non_temps = [c for c in names(data) if c ∉ [c_temps..., "profile_id"]]

    train_tensor_non_temp_x = zeros(Float32, (max_len_train, length(c_non_temps), n_train_profiles))
    train_tensor_temps_x = zeros(Float32, (max_len_train, length(c_input_temps), n_train_profiles))
    train_tensor_y = zeros(Float32, (max_len_train, length(target_cols), n_train_profiles))
    train_sample_weights = zeros(Float32, (max_len_train, n_train_profiles))

    test_tensor_non_temp_x = zeros(Float32, (max_len_test, length(c_non_temps), n_test_profiles))
    test_tensor_temps_x = zeros(Float32, (max_len_test, length(c_input_temps), n_test_profiles))
    test_tensor_y = zeros(Float32, (max_len_test, length(target_cols), n_test_profiles))
    test_sample_weights = zeros(Float32, (max_len_test, n_test_profiles));

    # fill in DataFrame information
    test_p_idx = 0
    train_p_idx = 0  # todo: something wrong with variable scope here
    @showprogress 0.5 "Generate tensors " for (pid, df) in pairs(gdf)
        if pid.profile_id ∈ test_set_pids
            test_p_idx += 1
            test_tensor_non_temp_x[1:nrow(df), :, test_p_idx] .= df[:, c_non_temps]
            test_tensor_temps_x[1:nrow(df), :, test_p_idx] .= df[:, c_input_temps]
            test_tensor_y[1:nrow(df), :, test_p_idx] .= df[:, target_cols]
            test_sample_weights[1:nrow(df), test_p_idx] .= 1
        else
            train_p_idx += 1
            train_tensor_non_temp_x[1:nrow(df), :, train_p_idx] .= df[:, c_non_temps]
            train_tensor_temps_x[1:nrow(df), :, train_p_idx] .= df[:, c_input_temps]
            train_tensor_y[1:nrow(df), :, train_p_idx] .= df[:, target_cols]
            train_sample_weights[1:nrow(df), train_p_idx] .= 1
        end
    end

    tbptt_len = 512

    train_vec_temps_x = [train_tensor_temps_x[i, :, :] for i in 1:size(train_tensor_temps_x, 1)]
    train_vec_non_temp_x = [train_tensor_non_temp_x[i, :, :] for i in 1:size(train_tensor_non_temp_x, 1)]
    train_vec_x = collect(zip(train_vec_non_temp_x, train_vec_temps_x))
    train_vec_y = [train_tensor_y[i, :, :] for i in 1:size(train_tensor_y, 1)]
    train_vec_sample_weights = [train_sample_weights[i, :] for i in 1:size(train_sample_weights, 1)]

    train_vec_chunked_x = []
    train_vec_chunked_y = []
    train_vec_chunked_w = []

    i = 0;
    while i * tbptt_len <= max_len_train
        push!(train_vec_chunked_x, train_vec_x[i * tbptt_len + 1:minimum(((i + 1) * tbptt_len + 1, max_len_train))])
        push!(train_vec_chunked_y, train_vec_y[i * tbptt_len + 1:minimum(((i + 1) * tbptt_len + 1, max_len_train))])
        push!(train_vec_chunked_w, train_vec_sample_weights[i * tbptt_len + 1:minimum(((i + 1) * tbptt_len + 1, max_len_train))])
        i += 1
    end

    test_vec_temps_x = [test_tensor_temps_x[i, :, :] for i in 1:size(test_tensor_temps_x, 1)]
    test_vec_non_temp_x = [test_tensor_non_temp_x[i, :, :] for i in 1:size(test_tensor_non_temp_x, 1)]
    test_vec_x = collect(zip(train_vec_non_temp_x, test_vec_temps_x))
    test_vec_y = [test_tensor_y[i, :, :] for i in 1:size(test_tensor_y, 1)]
    test_vec_sample_weights = [test_sample_weights[i, :] for i in 1:size(test_sample_weights, 1)];

    n_epochs = 100
    init_hidden = train_vec_y[1]
    m = Flux.Recur(TNNCell(length(c_non_temps) + length(c_input_temps),
                        length(c_temps),
                        length(target_cols),
                        init_hidden),
                init_hidden)
    ps = params(m)
    opt = ADAM(1e-3)

    function sample_weighted_loss(x::Vector{Tuple{Matrix{T},Matrix{T}}}, y::Vector{U}, w::Vector{V}) where {T,U,V}
        mean(Flux.Losses.mse(m(xi), yi, agg=z -> mean(wi' .* z)) for (xi, yi, wi) in zip(x, y, w)) 
    end

    # training
    pbar = Progress(n_epochs, desc="Training Epochs", start=1, showspeed=true)
    data_tup = zip(train_vec_chunked_x, train_vec_chunked_y, train_vec_chunked_w);
    for epoch in 1:n_epochs
        Flux.reset!(m)
        Flux.train!(sample_weighted_loss, ps, data_tup, opt)
        next!(pbar, showvalues=[(:epoch, epoch)])
    end

    # testing
    # todo

    # visualize
    # todo

end  # main

main()