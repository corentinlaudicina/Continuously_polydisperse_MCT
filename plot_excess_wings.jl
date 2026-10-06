#=
Figures for the susceptibilities χ''(ω), their excess wings and the KWW stretching of polydisperse
hard spheres. Each `plot_*` function makes one figure. None is called by default: uncomment the
call below the function you want.

    plot_relaxation_time()          τ_α vs distance ε to the critical point (average and per species)
    plot_ISF_susceptibility()       F̄_s(t) and rescaled χ̄''(ω) at fixed τ_α → Plots/ISF_susceptibility.pdf
    plot_β_KWW()                    β̄_KWW vs δ → Plots/KWW_exponent_avg_presentation.png
    plot_γ_a_b()                    MCT exponents γ, a and b vs δ
    plot_Fs_avg_polydispersity()    F̄_s(t) and species F_σ(t) for three δ → Plots/polydispersity_Fs_avg_uniform.pdf
    plot_species_susceptibility()   χ_σ''(ω) per species, one figure per δ → Plots/susceptibility_species_delta_<δ>.png
    plot_scaled_susceptibility_ϵ()  χ̄''(ω) approaching ϕ_c, with KWW and ω^a, ω^-b laws → Plots/susceptibility_epsilon.pdf
    plot_TTS()                      time-temperature superposition: F̄_s vs t/τ_α, one figure per δ
    plot_ISF()                      F̄_s(t) for the inverse-cubic distribution

Throughout, F̄_s and χ̄'' are averages over species, δ is the standard deviation of the diameters,
and f̄_c is the species-averaged tagged plateau at the peak of the summed structure factor.
=#

import Pkg
Pkg.activate(".")

using DelimitedFiles, JLD2, LaTeXStrings
using HDF5, CairoMakie
using LsqFit, SpecialFunctions, Roots, Measurements, Statistics

include("find_relaxation_time.jl")
include("calculate_susceptibility.jl")

# Minor ticks for axes holding log10 values: 9 ticks evenly spaced (in linear scale) between
# consecutive major ticks (not used below)
struct LogMinorTicks end


function MakieLayout.get_minor_tickvalues(::LogMinorTicks, tickvalues, vmin, vmax)
    vals = Float32[]
    for (lo, hi) in zip(@view(tickvalues[1:end-1]), @view(tickvalues[2:end]))
        interval = hi-lo
        steps = log10.(LinRange(10^lo, 10^hi, 11))
        append!(vals, steps[2:end-1])
    end
    vals
end

## The code below contains all functions necessary for the plots
## of the susceptibilities χ'', their wings and KWW stretching

"""
    find_critical_volume_fraction(δ, datafolder_crit)

Return the smallest volume fraction with a glass solution (sum of the collective non-ergodicity
parameter > 10⁻⁴), among the `fc_Ns_..._phi_<ϕ>.jld2` files in `datafolder_crit` whose diameters
have standard deviation exactly `δ`.
"""
function find_critical_volume_fraction(δ, datafolder_crit)

    files = readdir(datafolder_crit)
    println("$(length(files)) files found")
    files = files[contains.(files, "fc_Ns")]
    mydict = Dict{Float64, Float64}()
    for file in files
        f = jldopen(joinpath(datafolder_crit, file), "r")
        δ_f = std(f["run_params"]["D_arr"] ; corrected=false)

        if δ == δ_f
            fc = f["non_erg_param"]["fc_collective"]
            ϕ = parse(Float64, split(file, '_')[7][1:end-5])
            mydict[ϕ] = sum(sum(fc))
        else
            continue 
        end
        close(f)
    end
    sorted_ϕ_arr = sort(collect(keys(mydict)))

    for (i, ϕ) in enumerate(sorted_ϕ_arr)
        fc = mydict[ϕ]
        if fc > 0.0001
            return sorted_ϕ_arr[i]
        end
    end
end

"""
    find_crit_filename(dir_crit, ϕc)

Return the name of the file in `dir_crit` computed at volume fraction `ϕc`.
"""
function find_crit_filename(dir_crit, ϕc)

    for file in readdir(dir_crit)
        if parse(Float64, split(file, '_')[7][1:end-5]) == ϕc #) == ϕc
            return file
            break
        end
    end
end

"""
    find_fc_avg(dir_MCT_crit, filename)

Return the species-averaged tagged non-ergodicity parameter f̄_c^(s) at the peak of the summed
structure factor Σ_ij S_ij(k), from a critical-point file. Assumes the file was computed on the
grid `k = range(0.2, 39.8, length=100)`.
"""
function find_fc_avg(dir_MCT_crit, filename)
    f = jldopen(joinpath(dir_MCT_crit, filename), "r")

    Ns = f["run_params"]["Ns"]
    Sk = f["structure_factor"]
    k_array = range(0.2, 39.8, length=100)
    
    S_avg = zeros((length(k_array)))
    
    for kid in 1:length(k_array)
        val2 = 0.0
        for ns in 1:Ns 
            for nss in 1:Ns 
                val2 += Sk[kid][ns, nss]
            end
        end
        S_avg[kid] =  val2
    end

    k_peak_id = [i for i in eachindex(k_array) if S_avg[i] == maximum(S_avg)][1]

    fc_avg = 0.0

    for s in 1:Ns 
        fc = f["non_erg_param"]["f_c_$(s)"]
        fc_avg += fc[k_peak_id] 
    end
    fc_avg = fc_avg / Ns

    close(f)
    return fc_avg
end

"""
    find_fc_s(s, dir_MCT_crit, filename)

Return the tagged non-ergodicity parameter f_c^(s) of species `s` at the peak of the summed
structure factor Σ_ij S_ij(k) (same k-grid assumption as [`find_fc_avg`](@ref)).
"""
function find_fc_s(s, dir_MCT_crit, filename)
    f = jldopen(joinpath(dir_MCT_crit, filename), "r")

    Ns = f["run_params"]["Ns"]
    Sk = f["structure_factor"]
    k_array = range(0.2, 39.8, length=100)
    S_avg = zeros((length(k_array)))
    
    for kid in 1:length(k_array)
        val2 = 0.0
        for ns in 1:Ns 
            for nss in 1:Ns 
                val2 += Sk[kid][ns, nss]
            end
        end
        S_avg[kid] =  val2
    end

    k_peak_id = [i for i in eachindex(k_array) if S_avg[i] == maximum(S_avg)][1]

    fc_s = f["non_erg_param"]["f_c_$(s)"][k_peak_id]
    close(f)
    return fc_s
end

"""
    find_fc_s_peak_s(s, dir_MCT_crit, filename)

Return the tagged non-ergodicity parameter f_c^(s) of species `s` at the peak of its own partial
structure factor S_ss(k) (instead of the summed one), and plot S_ss(k).
"""
function find_fc_s_peak_s(s, dir_MCT_crit, filename)
    f = jldopen(joinpath(dir_MCT_crit, filename), "r")

    Sk = f["structure_factor"]

    k_array = range(0.2, 39.8, length=100)

    max_Sk = 0.0
    for kid in 1:length(k_array)
        if Sk[kid][s,s]>max_Sk
            max_Sk = Sk[kid][s,s]
        else
            continue 
        end
    end
    # @show max_Sk
    k_peak_id = [i for i in eachindex(k_array) if Sk[i][s,s] == max_Sk][1]
    # @show k_peak_id
    fc_s = f["non_erg_param"]["f_c_$(s)"][k_peak_id]
    close(f)

    f = Figure(resolution=(800,600))
    ax = Axis(f[1,1], title="s=$(s)", ylabel="S_{$(s)$(s)}(k)", xlabel="k")
    
    for kid in 1:length(k_array)
        scatter!(ax, [k_array[kid]], [Sk[kid][s,s]],
                 color="black")
    end

    display(f)
    @show fc_s
end

"""
    fit_stretched_exponential(fmin, fmax, t_arr, F, Fs_c)

Fit the KWW exponent β of `F(t) ≈ Fs_c exp(-(t/τ_α)^β)` and return it as `β ± σ_β` (a `Measurement`).

τ_α is fixed beforehand by `F(τ_α) = Fs_c/e`, and only β is fitted, in log scale, on the points
with `fmin < F < fmax`. If the fit returns its initial guess unchanged (relative change < 10⁻⁴),
it is restarted from a new random guess, at most 1000 times.
"""
function fit_stretched_exponential(fmin, fmax, t_arr, F, Fs_c)

    indices_to_fit = [i for i in eachindex(F) if fmin < F[i] < fmax]

    t_to_fit = t_arr[indices_to_fit]
    F_to_fit = F[indices_to_fit]
    
    τ_α = find_relaxation_time(t_arr,F; threshold=Fs_c/exp(1))
    
    powerlaw(t, p) = log10(Fs_c) .- log10(exp(1)).*(t./τ_α).^p[1]

    p0 = [rand()]
    maxcount = 1000
    count = 0
    fit_params = [0.4]
    fit_error = [0.0]
    while count < maxcount

        fit = curve_fit(powerlaw, t_to_fit, log10.(F_to_fit), p0)
        fit_params[1] = fit.param[1]
        # @show se = estimate_covar(fit)[1][1]
        fit_error[1] = sqrt(estimate_covar(fit)[1][1])
        if abs(fit_params[1]-p0[1])/p0[1] > 10.0^-4
            break 
        else 
            p0[1] = rand() 
        end

        count +=1
    end
    println("RETURNING KWW EXPONENT, $(count) ITERATIONS REQUIRED")
    return fit_params[1] ± fit_error[1]
end

"""
    fit_linear_function(x_arr, y_arr)

Fit `y = p₁ x + p₂` and return `[p₁, p₂]`, restarting from a new random guess if the fit returns
its initial slope unchanged (at most 1000 times).
"""
function fit_linear_function(x_arr, y_arr)

    linear_function(x, p) = p[1].*x .+ p[2]

    p0 = [rand(), rand()]
    maxcount = 1000
    count = 0
    fit_params = [-0.1, -0.1]
    while count < maxcount

        fit = curve_fit(linear_function, x_arr, y_arr, p0)
        fit_params[1] = fit.param[1]
        fit_params[2] = fit.param[2]

        if abs(fit_params[1]-p0[1])/p0[1] > 10.0^-4
            break 
        else 
            p0[1] = rand() 
        end

        count +=1
    end
    return fit_params
end

"""
    find_a_b_exponents(γ)

Return the MCT exponents `(a, b)` given γ, by solving Γ(1-a)²/Γ(1-2a) = Γ(1+b)²/Γ(1+2b)
with γ = 1/(2a) + 1/(2b), i.e. b = 1/(2(γ - 1/(2a))). Starts the root search at a = 0.9.
Errors on γ propagate when it is a `Measurement`.
"""
function find_a_b_exponents(γ)

    B(a) = 0.5*(γ-0.5/a)^-1

    f(x) = gamma(1-x)^2 / gamma(1-2*x) - gamma(1+B(x))^2/gamma(1+2*B(x))

    a = find_zero(f, (0.9±0.0))
    
    return a, B(a)
end

"""
    find_γ_exponent(δ, dir_sol)

Fit τ_α ∝ ε^(-γ) over all MCT solutions in `dir_sol` with polydispersity `δ`, where
ε = |ϕ - ϕ_c| and τ_α is defined by F̄_s(τ_α) = f̄_c/e (ϕ_c and f̄_c from `Data_Phase_Diagram`).
Returns `γ ± σ_γ` (a `Measurement`).
"""
function find_γ_exponent(δ, dir_sol)
    # dir_sol = "Data_sol_A3_epsilon"
    datafolder_crit = "Data_Phase_Diagram"
    files_mct_sol = String[]
    δ_mct_sol = String[]

    for file in readdir(dir_sol)
        if occursin("mctsol", file)
            push!(files_mct_sol, file)
        end
    end

    for sol in files_mct_sol
        f = h5open(joinpath(dir_sol, sol), "r")

        δ_sol = std(read(f["run_params"], "D_arr") ; corrected=false)

        if δ == δ_sol
            push!(δ_mct_sol, sol)
        end
        close(f)
    end
    δ_mct_sol = sort(δ_mct_sol)
    τ_arr = Float64[]
    ϕ_arr = Float64[]

    ϕc = find_critical_volume_fraction(δ, datafolder_crit)
    crit_filename = find_crit_filename(datafolder_crit, ϕc)
    @show datafolder_crit, crit_filename
    fc_avg = find_fc_avg(datafolder_crit, crit_filename)
    for sol in δ_mct_sol
        f = h5open(joinpath(dir_sol, sol), "r")
        Ns = read(f["run_params"], "Ns")
        ϕ = read(f["run_params"], "phi")
        t_array = read(f["F_tagged"], "t_array")
        Fs_avg = zeros(length(t_array))

        for s in 1:Ns 
            Fs_avg .+= read(f["F_tagged"], "F_s_$(s)") ./ Ns
        end

        τ_α = find_relaxation_time(t_array, Fs_avg ; threshold=fc_avg/exp(1))

        push!(ϕ_arr, ϕ)
        push!(τ_arr, τ_α)
        close(f)
    end

    ϵ_arr = abs.(ϕ_arr.- ϕc)

    linear_fit(x, p) = p[1] .+ p[2].*x
    p0 = [rand(), rand()]

    fit = curve_fit(linear_fit, log10.(ϵ_arr), log10.(τ_arr), p0)

    prefactor = fit.param[1]
    γ = fit.param[2]
    γ_cov = estimate_covar(fit)

    # fig = Figure()
    # ax = Axis(fig[1,1], yscale=log10, xscale=log10)


    # scatter!(ax, ϵ_arr, τ_arr)
    # lines!(ax, ϵ_arr, (10^prefactor).*ϵ_arr.^γ)
    # display(fig)
    return abs(γ) ± sqrt(γ_cov[2,2])
end

"""
    plot_relaxation_time()

Plot τ_α vs relative distance to the critical point ε = |ϕ - ϕ_c|/ϕ_c, for two polydispersities
(5th and second-to-last δ): the species average as a line, each species as red points.
"""
function plot_relaxation_time()

    dir_sol = "Data_sol_uniform_epsilon_paper"
    datafolder_crit = "Data_Phase_Diagram_Uniform"
    files_mct_sol_ϵ = readdir(dir_sol)
    
    δ_list = Any[]
    for file_sol in files_mct_sol_ϵ
        f = h5open(joinpath(dir_sol, file_sol), "r")
        δ = std(read(f["run_params"], "D_arr") ; corrected=false)
        Ns = read(f["run_params"], "Ns")
        close(f)
        if Ns == 10
            push!(δ_list, δ)
        end
    end
    δ_list = sort(union(δ_list))

    fig1 = Figure(resolution=(800,600))

    fontsize_theme = Theme(fontsize=20)
    set_theme!(fontsize_theme)
    Makie.theme(:fonts).:regular = "CMU Serif"

    fig1_ax1 = Axis(fig1[1,1], xscale=log10, yscale=log10, 
                    xlabel=L"$1/\epsilon$", ylabel=L"$\tau_{\alpha}$")

    for δ in [δ_list[5], δ_list[end-1]]
        τ_arr = Float64[]
        ϕ_arr = Float64[]

        ϕc = find_critical_volume_fraction(δ, datafolder_crit)
        crit_filename = find_crit_filename(datafolder_crit, ϕc)
        fc_avg = find_fc_avg(datafolder_crit, crit_filename)
        for sol in files_mct_sol_ϵ
            f = h5open(joinpath(dir_sol, sol), "r")
            
            if δ == std(read(f["run_params"], "D_arr") ; corrected=false)
                Ns = read(f["run_params"], "Ns")
                ϕ = read(f["run_params"], "phi")
                t_array = read(f["F_tagged"], "t_array")
                Fs_avg = zeros(length(t_array))

                for s in 1:Ns 
                    Fs_avg .+= read(f["F_tagged"], "F_s_$(s)") ./ Ns
                    fc_s = find_fc_s(s, datafolder_crit, crit_filename)
                    τ_αs = find_relaxation_time(t_array, read(f["F_tagged"], "F_s_$(s)") ; threshold=fc_s/exp(1))

                    scatter!(fig1_ax1, [abs(ϕ-ϕc)/ϕc], [τ_αs], color=("red", 0.5))
                end

                τ_α = find_relaxation_time(t_array, Fs_avg ; threshold=fc_avg/exp(1))

                push!(ϕ_arr, ϕ)
                push!(τ_arr, τ_α)
            else
                continue 
            end
            close(f)
        end

        ϵ_arr = abs.(ϕ_arr.- ϕc)./ϕc
        @show ϵ_arr, τ_arr
        lines!(fig1_ax1, ϵ_arr, τ_arr)

    end

    display(fig1)
end
# plot_relaxation_time()

"""
    plot_ISF_susceptibility()

At fixed τ_α (`Data_sol_uniform_fixed_tau_alpha`), plot (a) F̄_s(t) and (b) χ̄''(ω)/f̄_c vs ω/ω̄₀,
where ω̄₀ is the position of the minimum of χ̄'' for 10⁻⁶ < ω < 10⁻². Shows every other Ns = 10
run, with the monodisperse run (Ns = 1) in black. Saves `Plots/ISF_susceptibility.pdf`.
"""
function plot_ISF_susceptibility()

    dir = "Data_sol_uniform_fixed_tau_alpha"
    dir_crit = "Data_Phase_Diagram_Uniform"
    files_mct_sol = String[]

    for file in readdir(dir)
        if occursin("mctsol", file)
            push!(files_mct_sol, file)
        end
    end

    fig2 = Figure(resolution=(850,400))

    fontsize_theme = Theme(fontsize=25)
    set_theme!(fontsize_theme)
    Makie.theme(:fonts).:regular = "CMU Serif"

    fig2_ax1 = Axis(fig2[1:2, 1], xtickalign=0, ytickalign=0, ylabel=L"$\overline{F^{(s)}}(k,\, t)$", 
               xlabel=L"$t$", xscale=log10, xminorticksvisible=false, yminorticksvisible=true,
            #    xminorticks = IntervalsBetween(9),xminorticksvisible=true,xminortickalign=1,
               limits=(10.0^-3, 10.0^11, 0, 1), xgridvisible=false, ygridvisible=false)

    fig2_ax2 = Axis(fig2[1:2, 2], xtickalign = 0, ytickalign=0, xminorticksvisible=false, yminorticksvisible=true, yaxisposition=:left, 
               ylabel=L"$\overline{\chi}\"(k,\, \omega)/\overline{f^{(s)}}(k)$", 
               xlabel=L"$\omega / \overline{\omega}_0$", xscale=log10, yscale=log10, 
            #    xminorticks = IntervalsBetween(9),xminorticksvisible=true,xminortickalign=1,yminorticks = IntervalsBetween(9),yminorticksvisible=true,yminortickalign=1,
               limits=(10.0^-6, 10.0^6, 5*10.0^-3, 10.0^-0.3), xgridvisible=false, ygridvisible=false)

    text!(fig2_ax1, 10.0^9, 0.90, text=L"\text{(a)}")
    text!(fig2_ax2, 2*10.0^-6, 10^-0.5, text=L"\text{(b)}")

    colmap = cgrad(:matter, length(files_mct_sol)+1, categorical = true)
    col=2
    count = 0

    δ_list = Any[]
    for file_sol in files_mct_sol
        f = h5open(joinpath(dir, file_sol), "r")
        δ = std(read(f["run_params"], "D_arr") ; corrected=false)
        Ns = read(f["run_params"], "Ns")
        close(f)
        if Ns == 10
            push!(δ_list, δ)
        end
    end
    for file_sol in files_mct_sol

        f = h5open(joinpath(dir, file_sol), "r")
        t_arr = read(f["F_tagged"],"t_array")
        ω_arr = read(f["relaxation_spectrum"],"omega_arr")
        Ns = read(f["run_params"], "Ns")
        δ = std(read(f["run_params"], "D_arr") ; corrected=false)

        χs_avg = zeros(length(ω_arr))
        Fs_avg = zeros(length(t_arr))

        for s in 1:Ns
            Fs_avg .+= read(f["F_tagged"],"F_s_$(s)") ./ Ns
            χs_avg .+= read(f["relaxation_spectrum"], "chi_s_$(s)")./Ns
        end
        
        close(f)

        ϕhi = find_critical_volume_fraction(δ, dir_crit)
        crit_filename = find_crit_filename(dir_crit, ϕhi)
        fc_avg = find_fc_avg(dir_crit, crit_filename)
        if Ns == 10 && col%2 == 0 

            scatter!(fig2_ax1, t_arr[2:10:end], Fs_avg[2:10:end], 
                    color = colmap[col],
                    markerstrokewidth = 0.0, 
                    label=L"$\delta=%$(round(δ, digits=2))$")

            indices_min = [i for i in eachindex(ω_arr) if 10.0^-6 < ω_arr[i] < 10.0^-2]
            chi_avg_0 = minimum(χs_avg[indices_min])
            id = 1 

            for val in χs_avg
                if val == chi_avg_0
                    break 
                else
                    id += 1
                end
            end

            scatter!(fig2_ax2, ω_arr[1:2:end]./ω_arr[id], χs_avg[1:2:end]./fc_avg,
                    markerstrokewidth = 0.0, 
                    color = colmap[col])
            
        end
        if Ns == 1

            lines!(fig2_ax1, t_arr[2:end], Fs_avg[2:end], 
                color = ("black", 0.5), 
                alpha=0.5)
            
            indices_min = [i for i in eachindex(ω_arr) if 10.0^-6 < ω_arr[i] < 10.0^-2]
            chi_avg_max = minimum(χs_avg[indices_min])
            id = 1 

            for val in χs_avg
                if val == chi_avg_max
                    break 
                else
                    id += 1
                end
            end
    
            lines!(fig2_ax2, ω_arr[2:end]./ω_arr[id], 
                χs_avg[2:end]./fc_avg, 
                color = ("black", 0.5))
                
        end
        col +=1
    end

    axislegend(fig2_ax1, position = :lb, labelsize=20)
    save("Plots/ISF_susceptibility.pdf", fig2)
    display(fig2)
end

# plot_ISF_susceptibility()

"""
    plot_β_KWW()

Plot the species-averaged KWW exponent β̄_KWW (with error bars) vs δ, fitted at the state point
closest to ϕ_c for each δ (`Data_sol_uniform_epsilon`), using the window f̄_c/10 < F̄_s < f̄_c/2.
Per-species exponents are also fitted for two δ, but their plots are commented out.
Saves `Plots/KWW_exponent_avg_presentation.png`.
"""
function plot_β_KWW()
    dir_ϵ = "Data_sol_uniform_epsilon"
    dir_crit = "Data_Phase_Diagram_Uniform"
    files_mct_sol_ϵ = readdir(dir_ϵ)
    δ_list = Any[]
    for file_sol in files_mct_sol_ϵ
        f = h5open(joinpath(dir_ϵ, file_sol), "r")
        δ = std(read(f["run_params"], "D_arr") ; corrected=false)
        Ns = read(f["run_params"], "Ns")
        close(f)
        if Ns == 10
            push!(δ_list, δ)
        end
    end
    δ_list = sort(union(δ_list))
    @show δ_list
    colmap = cgrad(:matter, length(δ_list)+1, categorical = true)

    max_ϕ_list = Float64[]
    for δ in δ_list
        list_ϕ = Float64[]
        for file in files_mct_sol_ϵ
            f = h5open(joinpath(dir_ϵ, file), "r")
            δ_sol = std(read(f["run_params"], "D_arr") ; corrected=false)

            if δ == δ_sol
                ϕ = parse(Float64, split(file, '_')[7][1:end-5])
                push!(list_ϕ, ϕ)
            end
            close(f)
        end
        push!(max_ϕ_list, maximum(list_ϕ))
    end

        # fig1 = Figure(resolution=(800,600))
        fig1 = Figure(resolution=(600,600))
        fontsize_theme = Theme(fontsize=25)
        set_theme!(fontsize_theme)
        Makie.theme(:fonts).:regular = "CMU Serif"
        
        fig1_ax1 = Axis(fig1[1,1], 
                        xlabel=L"$\delta$", ylabel=L"$\overline{\beta}_{\text{KWW}}$", xgridvisible=false, ygridvisible=false
                       )
        # fig1_ax2 = Axis(fig1[1,2], 
        #                 xlabel=L"$D$", 
        #                 ylabel=L"$\beta_{\text{KWW}}$",
        #                 yaxisposition=:right, xgridvisible=false, ygridvisible=false)
        # fig1_ax3 = Axis(fig1[2,2], 
        #                 xlabel=L"$D$", ylabel=L"$\beta_{\text{KWW}}$",
        #                 yaxisposition=:right, xgridvisible=false, ygridvisible=false)
    

    for ϕmax in max_ϕ_list
        for file in files_mct_sol_ϵ
            if parse(Float64, split(file, '_')[7][1:end-5]) == ϕmax
                f = h5open(joinpath(dir_ϵ, file), "r")
                t_arr = read(f["F_tagged"],"t_array")
                Ns = read(f["run_params"], "Ns")
                D_arr = read(f["run_params"], "D_arr")
                δ = std(read(f["run_params"], "D_arr") ; corrected=false)
        
                Fs_avg = zeros(length(t_arr))
                
                ϕhi = find_critical_volume_fraction(δ, dir_crit)
                crit_filename = find_crit_filename(dir_crit, ϕhi)
                fc_avg = find_fc_avg(dir_crit, crit_filename)
                fc_max = fc_avg/2
                fc_min = fc_avg/10

                for s in 1:Ns
                    Fs_avg .+= read(f["F_tagged"],"F_s_$(s)") ./ Ns

                    if δ == δ_list[5]
                        Fs = read(f["F_tagged"],"F_s_$(s)")
                        fc_s = find_fc_s(s,dir_crit, crit_filename)
                        fc_s_min = fc_s/10
                        fc_s_max = fc_s/2
                        if fc_s > 10.0^-2
                            β_s = fit_stretched_exponential(fc_s_min, fc_s_max, t_arr, Fs, fc_s).val
                            err_β_s = fit_stretched_exponential(fc_s_min, fc_s_max, t_arr, Fs, fc_s).err
                            # scatter!(fig1_ax2, [D_arr[s]], [β_s],
                            # markerstrokewidth=0.0, color=colmap[5])
                            # errorbars!(fig1_ax2, [D_arr[s]], [β_s], [err_β_s],
                            # whiskerwidth = 10, color=colmap[5])

                        end
                    end 

                    if δ == δ_list[end-3]
                        Fs = read(f["F_tagged"],"F_s_$(s)")
                        fc_s = find_fc_s(s,dir_crit, crit_filename)
                        fc_s_min = fc_s/10
                        fc_s_max = fc_s/2
                        if fc_s > 10.0^-2
                            β_s = fit_stretched_exponential(fc_s_min, fc_s_max, t_arr, Fs, fc_s).val
                            err_β_s = fit_stretched_exponential(fc_s_min, fc_s_max, t_arr, Fs, fc_s).err
                            # scatter!(fig1_ax3, [D_arr[s]], [β_s],
                            # markerstrokewidth=0.0, color=colmap[end-1])
                            # errorbars!(fig1_ax3, [D_arr[s]], [β_s], [err_β_s],
                            # whiskerwidth = 10, color=colmap[end-1])
                        end
                    end 

                end

                if fc_avg > 10.0^-2
                    β_KWW_avg = fit_stretched_exponential(fc_min, fc_max, t_arr, Fs_avg, fc_avg).val
                    err_β_KWW_avg = fit_stretched_exponential(fc_min, fc_max, t_arr, Fs_avg, fc_avg).err
                    
                    scatter!(fig1_ax1, [δ], [β_KWW_avg], 
                            markerstrokewidth=0.0, color=colmap[end]) 
                    errorbars!(fig1_ax1, [δ], [β_KWW_avg], [err_β_KWW_avg],
                            whiskerwidth = 10, color=colmap[end])
             
                    if δ == δ_list[5]
                        # hlines!(fig1_ax2, β_KWW_avg, 
                        #         linestyle=:dash, color=("black", 0.75))
                    end
                    if δ == δ_list[end-1]
                        # hlines!(fig1_ax3, β_KWW_avg, 
                        #         linestyle=:dash, color=("black", 0.75))
                    end

                end
                close(f)
            end
        end
    end
    # text!(fig1_ax2, 1.12, 0.655, text=L"$\delta = %$(round(δ_list[5], digits=2))$")
    # text!(fig1_ax3, 1.2, 0.60, text=L"$\delta = %$(round(δ_list[end-3], digits=2))$")


    # text!(fig1_ax1, 0.47, 0.75, text=L"\text{(a)}")
    # text!(fig1_ax2, 0.6, 0.77, text=L"\text{(b)}")
    # text!(fig1_ax3, 0.42, 0.745, text=L"\text{(c)}")

    display(fig1)
    save("Plots/KWW_exponent_avg_presentation.png",fig1)
end
# plot_β_KWW()

"""
    plot_γ_a_b()

Plot the MCT exponents γ (from [`find_γ_exponent`](@ref)), a and b (from
[`find_a_b_exponents`](@ref)) with error bars vs δ, for the uniform distribution.
"""
function plot_γ_a_b()
    # dir_ϵ_inverse_cubic = "Data_sol_A3_epsilon"
    # # dir_crit = "Data_Phase_Diagram"
    # files_mct_sol_ϵ_inverse_cubic = readdir(dir_ϵ_inverse_cubic)
    # δ_list_inverse_cubic = Float64[]
    # for file_sol in files_mct_sol_ϵ_inverse_cubic
    #     f = h5open(joinpath(dir_ϵ_inverse_cubic, file_sol), "r")
    #     δ = std(read(f["run_params"], "D_arr") ; corrected=false)
    #     Ns = read(f["run_params"], "Ns")
    #     close(f)
    #     if Ns == 10
    #         push!(δ_list_inverse_cubic, δ)
    #     end
    # end
    # δ_list_inverse_cubic = sort(union(δ_list_inverse_cubic))

    # γ_list_inverse_cubic = Float64[]
    # γ_err_list_inverse_cubic = Float64[]
    # a_list_inverse_cubic = Float64[]
    # b_list_inverse_cubic = Float64[]
    # a_err_list_inverse_cubic = Float64[]
    # b_err_list_inverse_cubic = Float64[]


    dir_ϵ_uniform = "Data_sol_uniform_epsilon_paper"
    # dir_crit = "Data_Phase_Diagram"
    files_mct_sol_ϵ_uniform = readdir(dir_ϵ_uniform)
    δ_list_uniform = Float64[]
    for file_sol in files_mct_sol_ϵ_uniform
        f = h5open(joinpath(dir_ϵ_uniform, file_sol), "r")
        δ = std(read(f["run_params"], "D_arr") ; corrected=false)
        Ns = read(f["run_params"], "Ns")
        close(f)
        if Ns == 10
            push!(δ_list_uniform, δ)
        end
    end
    δ_list_uniform = sort(union(δ_list_uniform))

    γ_list_uniform = Float64[]
    γ_err_list_uniform = Float64[]
    a_list_uniform = Float64[]
    b_list_uniform = Float64[]
    a_err_list_uniform = Float64[]
    b_err_list_uniform = Float64[]

    colmap = cgrad(:matter, 2, categorical = true)

    fig = Figure(resolution=(800,600))

    fontsize_theme = Theme(fontsize=25)
    set_theme!(fontsize_theme)
    Makie.theme(:fonts).:regular = "CMU Serif"

    fig_ax1 = Axis(fig[1:2,1], 
                    xlabel=L"$\delta$", 
                    ylabel=L"$\gamma$", xgridvisible=false, ygridvisible=false)

    fig_ax2 = Axis(fig[1,2], 
                    xlabel=L"$\delta$", 
                    ylabel=L"$a$",
                    yaxisposition=:right, xgridvisible=false, ygridvisible=false)
    fig_ax3 = Axis(fig[2,2], 
                    xlabel=L"$\delta$", 
                    ylabel=L"$b$",
                    yaxisposition=:right, xgridvisible=false, ygridvisible=false)
    text!(fig_ax1, 0.06, 2.745, text=L"\text{(a)}")
    text!(fig_ax2, 0.48, 0.308, text=L"\text{(b)}")
    text!(fig_ax3, 0.48, 0.572, text=L"\text{(c)}")

    # for δ in δ_list_inverse_cubic
    #     γ = find_γ_exponent(δ, dir_ϵ_inverse_cubic)

    #     push!(γ_list_inverse_cubic, γ.val)
    #     push!(γ_err_list_inverse_cubic, γ.err)
    #     a,b = find_a_b_exponents(γ)
    #     push!(a_list_inverse_cubic, a.val)
    #     push!(b_list_inverse_cubic, b.val) 
    #     push!(a_err_list_inverse_cubic, a.err)
    #     push!(b_err_list_inverse_cubic, b.err) 

    # end

    # errorbars!(fig_ax1, δ_list_inverse_cubic, γ_list_inverse_cubic, γ_err_list_inverse_cubic,
    #            whiskerwidth = 10, color=colmap[end])
    # errorbars!(fig_ax2, δ_list_inverse_cubic, a_list_inverse_cubic, a_err_list_inverse_cubic,
    #            whiskerwidth = 10, color=colmap[end])
    # errorbars!(fig_ax3, δ_list_inverse_cubic, b_list_inverse_cubic, b_err_list_inverse_cubic,
    #            whiskerwidth = 10, color=colmap[end])
    # scatter!(fig_ax1, δ_list_inverse_cubic, γ_list_inverse_cubic, color=colmap[end])
    # scatter!(fig_ax2, δ_list_inverse_cubic, a_list_inverse_cubic, color=colmap[end])
    # scatter!(fig_ax3, δ_list_inverse_cubic, b_list_inverse_cubic, color=colmap[end])


    for δ in δ_list_uniform
        γ = find_γ_exponent(δ, dir_ϵ_uniform)

        push!(γ_list_uniform, γ.val)
        push!(γ_err_list_uniform, γ.err)
        a,b = find_a_b_exponents(γ)
        push!(a_list_uniform, a.val)
        push!(b_list_uniform, b.val) 
        push!(a_err_list_uniform, a.err)
        push!(b_err_list_uniform, b.err) 

    end

    errorbars!(fig_ax1, δ_list_uniform, γ_list_uniform, γ_err_list_uniform,
               whiskerwidth = 10, color=colmap[end])
    errorbars!(fig_ax2, δ_list_uniform, a_list_uniform, a_err_list_uniform,
               whiskerwidth = 10, color=colmap[end])
    errorbars!(fig_ax3, δ_list_uniform, b_list_uniform, b_err_list_uniform,
               whiskerwidth = 10, color=colmap[end])
    scatter!(fig_ax1, δ_list_uniform, γ_list_uniform, color=colmap[end])
    scatter!(fig_ax2, δ_list_uniform, a_list_uniform, color=colmap[end])
    scatter!(fig_ax3, δ_list_uniform, b_list_uniform, color=colmap[end])


    display(fig)
    # save("Plots/MCT_exponents_polydispersity.pdf", fig)
end

# plot_γ_a_b()

"""
    plot_Fs_avg_polydispersity()

At fixed τ_α, plot F̄_s(t) (solid) and the per-species F_σ(t) (dashed) vs log t for three
polydispersities (2nd, 5th and 4th-to-last δ, in file order). Saves
`Plots/polydispersity_Fs_avg_uniform.pdf`.
"""
function plot_Fs_avg_polydispersity()

    dir = "Data_sol_uniform_fixed_tau_alpha"

    files_mct_sol = String[]
    
    for file in readdir(dir)
        if occursin("mctsol", file)
            push!(files_mct_sol, file)
        end
    end
    
    δ_list = Any[]
    for file_sol in files_mct_sol
        f = h5open(joinpath(dir, file_sol), "r")
        δ = std(read(f["run_params"], "D_arr") ; corrected=false)
        Ns = read(f["run_params"], "Ns")
        close(f)
        if Ns == 10
            push!(δ_list, δ)
        end
    end
    
    fig1 = Figure(resolution=(1250,500))

    fig2 = Figure(resolution=(1200,400))

    fontsize_theme = Theme(fontsize=22)
    set_theme!(fontsize_theme)

    Makie.theme(:fonts).:regular = "CMU Serif"

    ax1 = Axis(fig2[1, 1], xtickalign=0, ytickalign=0, ylabel=L"$\overline{F^{(s)}}(k,\, t), \, F_{\sigma}^{(s)}(k,\, t)$", xlabel=L"$\log(t)$",xticks=-2:2:11,yticks=LinRange(0.0,1.0,5), limits=(-2, 11, 0, 1), 
    title = L"$\delta = %$(round(δ_list[2], digits=2))$", xgridvisible=false, ygridvisible=false)

    ax2 = Axis(fig2[1, 2], xtickalign=0, ytickalign=0, xticks=-2:2:11,yticks=LinRange(0.0,1.0,5), ylabel=L"$\overline{F^{(s)}}(k,\, t), \, F_{\sigma}^{(s)}(k,\, t)$", xlabel=L"$\log(t)$", limits=(-2, 11, 0, 1),
    title = L"$\delta = %$(round(δ_list[5], digits=2))$", xgridvisible=false, ygridvisible=false)

    ax3 = Axis(fig2[1, 3], xtickalign=0, ytickalign=0, ylabel=L"$\overline{F^{(s)}}(k,\, t), \, F_{\sigma}^{(s)}(k,\, t)$", xlabel=L"$\log(t)$",xticks=-2:2:11,yticks=LinRange(0.0,1.0,5), limits=(-2, 11, 0, 1),
    title = L"$\delta = %$(round(δ_list[end-3], digits=2))$", xgridvisible=false, ygridvisible=false)

    # fig1_ax1 = Axis(fig1[1, 1], ylabel=L"$\overline{\chi}_s''(k,\, t), \chi_{\sigma}^{(s)}''(k, t)$", xlabel=L"$\omega$", xscale=log10, yscale=log10, limits=(10.0^-11, 10.0^2, 5*10.0^-3, 0.5),
    # title = L"$\delta = %$(round(δ_list[div(length(δ_list),2)], digits=3))$")

    # fig1_ax2 = Axis(fig1[1, 2], ylabel=L"$\overline{\chi}_s''(k,\, t), \chi_{\sigma}^{(s)}''(k, t)$", xlabel=L"$\omega$", xscale=log10, yscale=log10, limits=(10.0^-11, 10.0^2, 5*10.0^-3, 0.5),
    # title = L"$\delta = %$(round(δ_list[div(length(δ_list),2)], digits=3))$")

    # fig1_ax3 = Axis(fig1[1, 3], ylabel=L"$\overline{\chi}_s''(k,\, t), \chi_{\sigma}^{(s)}''(k, t)$", xlabel=L"$\omega$", xscale=log10, yscale=log10, limits=(10.0^-11, 10.0^2, 5*10.0^-3, 0.5),
    # title = L"$\delta = %$(round(δ_list[div(length(δ_list),2)], digits=3))$")
    x_pos = 9.3
    y_pos = 0.88
    text!(ax1, x_pos, y_pos, text=L"\text{(a)}")
    text!(ax2, x_pos, y_pos, text=L"\text{(b)}")
    text!(ax3, x_pos, y_pos, text=L"\text{(c)}")

    alpha = 0.5
    hlines!(ax1, [1/exp(1)], color=("black", alpha))
    hlines!(ax2, [1/exp(1)], color=("black", alpha))
    hlines!(ax3, [1/exp(1)], color=("black", alpha))

    colmap = cgrad(:matter, length(files_mct_sol)+1, categorical = true)
    col = 2

    for file_sol in files_mct_sol

        f = h5open(joinpath(dir, file_sol), "r")
        t_arr = read(f["F_tagged"],"t_array")
        # ω_arr = read(f["relaxation_spectrum"],"omega_arr")
        Ns = read(f["run_params"], "Ns")
        δ = std(read(f["run_params"], "D_arr") ; corrected=false)

        if δ == δ_list[2] 
            @show read(f["run_params"], "phi")
            Fs_avg = zeros(length(t_arr))
            # chi_avg = zeros(length(ω_arr))

            for s in 1:Ns
                Fs_avg .+= read(f["F_tagged"],"F_s_$(s)") ./ Ns
                # chi_avg .+= read(f["relaxation_spectrum"],"chi_s_$(s)") ./ Ns
                lines!(ax1, log10.(t_arr[2:end]), read(f["F_tagged"],"F_s_$(s)")[2:end], 
                    color=(colmap[col], 0.5), linestyle=:dash)

                # lines!(fig1_ax1, ω_arr, read(f["relaxation_spectrum"],"chi_s_$(s)"), 
                    # color=(colmap[col], 0.5), linestyle=:dash)

            end

            lines!(ax1, log10.(t_arr[2:end]), Fs_avg[2:end], color=colmap[col])
            # lines!(fig1_ax1, ω_arr, chi_avg, color=colmap[col])

            col += 1
        elseif δ == δ_list[5]
            @show read(f["run_params"], "phi")
            Fs_avg = zeros(length(t_arr))
            # chi_avg = zeros(length(ω_arr))

            for s in 1:Ns
                Fs_avg .+= read(f["F_tagged"],"F_s_$(s)") ./ Ns
                # chi_avg .+= read(f["relaxation_spectrum"],"chi_s_$(s)") ./ Ns

                lines!(ax2, log10.(t_arr[2:end]), read(f["F_tagged"],"F_s_$(s)")[2:end], 
                color=(colmap[col], 0.5), linestyle=:dash)

                # lines!(fig1_ax2, ω_arr, read(f["relaxation_spectrum"],"chi_s_$(s)"), 
                # color=(colmap[col], 0.5), linestyle=:dash)

            end

            lines!(ax2, log10.(t_arr[2:end]), Fs_avg[2:end], color=colmap[col])
            # lines!(fig1_ax2, ω_arr, chi_avg, color=colmap[col])

            col += 1

        elseif δ == δ_list[end-3]
            @show read(f["run_params"], "phi")
            
            Fs_avg = zeros(length(t_arr))
            # chi_avg = zeros(length(ω_arr))

            for s in 1:Ns
                Fs_avg .+= read(f["F_tagged"],"F_s_$(s)") ./ Ns
                # chi_avg .+= read(f["relaxation_spectrum"],"chi_s_$(s)") ./ Ns

                lines!(ax3, log10.(t_arr[2:end]), read(f["F_tagged"],"F_s_$(s)")[2:end], 
                color=(colmap[col], 0.5), linestyle=:dash)

                # lines!(fig1_ax3, ω_arr, read(f["relaxation_spectrum"],"chi_s_$(s)"), 
                # color=(colmap[col], 0.5), linestyle=:dash)

            end

            lines!(ax3, log10.(t_arr[2:end]), Fs_avg[2:end], color=colmap[col])
            # lines!(fig1_ax3, ω_arr, chi_avg, color=colmap[col])

            col += 1

        else
            col += 1
        end    
        close(f)
    end
    display(fig2)
    # display(fig1)
    save("Plots/polydispersity_Fs_avg_uniform.pdf", fig2)
    # save("Plots/polydispersity_chi_avg.pdf", fig1)
end

# plot_Fs_avg_polydispersity()

"""
    plot_species_susceptibility()

At fixed τ_α, plot the per-species susceptibilities χ_σ''(ω) (dashed) and their average (points),
one figure per file, saved as `Plots/susceptibility_species_delta_<δ>.png`.
"""
function plot_species_susceptibility()

    dir = "Data_sol_uniform_fixed_tau_alpha"

    files_mct_sol = String[]
    
    for file in readdir(dir)
        if occursin("mctsol", file)
            push!(files_mct_sol, file)
        end
    end
    
    δ_list = Any[]
    for file_sol in files_mct_sol
        f = h5open(joinpath(dir, file_sol), "r")
        δ = std(read(f["run_params"], "D_arr") ; corrected=false)
        Ns = read(f["run_params"], "Ns")
        close(f)
        if Ns == 10
            push!(δ_list, δ)
        end
    end
    
    # fig1 = Figure(resolution=(600,600))

    # Makie.theme(:fonts).:regular = "CMU Serif"

    # fig1_ax1 = Axis(fig1[1, 1], ylabel=L"$\overline{\chi}_s''(k,\, t), \chi_{\sigma}^{(s)}''(k, t)$", xlabel=L"$\omega$", xscale=log10, yscale=log10, limits=(10.0^-11, 10.0^2, 5*10.0^-3, 0.5),
    # title = L"$\delta = %$(round(δ_list[div(length(δ_list),2)], digits=3))$")

    # text!(ax1, 10.0^9, 0.90, text=L"\text{(a)}")
    # text!(ax2, 10.0^9, 0.90, text=L"\text{(b)}")
    # text!(ax3, 10.0^9, 0.90, text=L"\text{(c)}")

    colmap = cgrad(:matter, length(files_mct_sol)+1, categorical = true)
    col = 2
    for file_sol in files_mct_sol

        f = h5open(joinpath(dir, file_sol), "r")
        ω_arr = read(f["relaxation_spectrum"],"omega_arr")
        Ns = read(f["run_params"], "Ns")
        δ = std(read(f["run_params"], "D_arr") ; corrected=false)


        fig1 = Figure(resolution=(600,600))
        Makie.theme(:fonts).:regular = "CMU Serif"

        fig1_ax1 = Axis(fig1[1, 1], ylabel=L"$\overline{\chi}_s''(k,\, t), \chi_{\sigma}^{(s)}''(k, t)$", xlabel=L"$\omega$", xscale=log10, yscale=log10, limits=(10.0^-11, 10.0^2, 5*10.0^-3, 0.5),
        title = L"$\delta = %$(round(δ, digits=3))$")
    
    

        chi_avg = zeros(length(ω_arr))

        for s in 1:Ns
            chi_avg .+= read(f["relaxation_spectrum"],"chi_s_$(s)") ./ Ns

            lines!(fig1_ax1, ω_arr, read(f["relaxation_spectrum"],"chi_s_$(s)"), 
                color=(colmap[col], 0.8), linestyle=:dash)

        end

        scatter!(fig1_ax1, ω_arr[1:end], chi_avg[1:end], color=colmap[col],
                markerstrokewidth=0.0)

        close(f)
        display(fig1)
        save("Plots/susceptibility_species_delta_$(δ).png", fig1)
        col += 1
    end
    # display(fig2)
    # display(fig1)
    # save("Plots/polydispersity_Fs_avg.pdf", fig2)
    # save("Plots/polydispersity_chi_avg.pdf", fig1)
end
# plot_species_susceptibility()

"""
    plot_scaled_susceptibility_ϵ()

Susceptibilities approaching the transition for two polydispersities (5th and 4th-to-last δ, columns):
  - top row (`fig_ax1`, `fig_ax2`): χ̄''(ω) at every state with |ϕ - ϕ_c| < 0.01, each with the KWW
    susceptibility for the β fitted at the closest state and its own τ_α;
  - bottom row (`fig_ax*_top`): the closest state, with the per-species χ_σ'', the KWW fit and the
    ω^a and ω^(-b) power laws.
Saves `Plots/susceptibility_epsilon.pdf`.
"""
function plot_scaled_susceptibility_ϵ()
    dir = "Data_sol_uniform_epsilon_paper"
    dir_crit = "Data_Phase_Diagram_Uniform"

    files_mct_sol = String[]

    for file in readdir(dir)
        if occursin("mctsol", file)
            push!(files_mct_sol, file)
        end
    end

    δ_list = Any[]
    for file_sol in files_mct_sol
        f = h5open(joinpath(dir, file_sol), "r")
        δ = std(read(f["run_params"], "D_arr") ; corrected=false)
        Ns = read(f["run_params"], "Ns")

        close(f)
        if Ns == 10
            push!(δ_list, δ)
        end
    end

    δ_list = sort(union(δ_list))

    colmap = cgrad(:matter, length(δ_list)+1, categorical = true)

    δ_list_to_plot = [δ_list[5], δ_list[end-3]]
    fig = Figure(resolution=(1000,800))

    fontsize_theme = Theme(fontsize=30)
    set_theme!(fontsize_theme)
    Makie.theme(:fonts).:regular = "CMU Serif"

    fig_ax1_top = Axis(fig[2, 1], xtickalign = 0, ytickalign=0, ylabel=L"$\overline{\chi}\"(k,\, \omega)$", 
                   xticklabelsvisible =true, xlabel=L"$\omega$", 
                #    xminorticks = IntervalsBetween(9),xminorticksvisible=true,xminortickalign=1,yminorticks = IntervalsBetween(9),yminorticksvisible=true,yminortickalign=1,
                   xscale=log10, yscale=log10,
                   limits=(10.0^-14, 10.0^3, 10.0^-3, 10.0^-0.3), xgridvisible=false, ygridvisible=false)
                #    , xlabelsize=20, ylabelsize=20)

    fig_ax2_top = Axis(fig[2, 2], xtickalign = 0, ytickalign=0, yaxisposition=:left, 
                #    xminorticks = IntervalsBetween(9),xminorticksvisible=true,xminortickalign=1,yminorticks = IntervalsBetween(9),yminorticksvisible=true,yminortickalign=1,
                   ylabel=L"$\overline{\chi}\"(k,\, \omega)$", xlabel=L"$\omega$", 
                   xticklabelsvisible =true, xscale=log10, yscale=log10, 
                   limits=(10.0^-14, 10.0^3, 10.0^-3, 10.0^-0.3), xgridvisible=false, ygridvisible=false)
                #    , xlabelsize=20, ylabelsize=20)
    
    fig_ax1 = Axis(fig[1, 1], xtickalign=1, ytickalign=0, ylabel=L"$\overline{\chi}\"(k,\, \omega)$", 
                   xscale=log10, yscale=log10,
                   xticklabelsvisible=false,
                #    xminorticks = IntervalsBetween(9),xminorticksvisible=true,xminortickalign=1,yminorticks = IntervalsBetween(9),yminorticksvisible=true,yminortickalign=1,
                   limits=(10.0^-14, 10.0^3, 10.0^-3, 10.0^-0.3), 
                   title=L"$\delta = %$(round(δ_list_to_plot[1], digits=2))$", xgridvisible=false, ygridvisible=false)
                #    , xlabelsize=20, ylabelsize=20)

    fig_ax2 = Axis(fig[1, 2], xtickalign = 1, ytickalign=0, yaxisposition=:left, 
                   ylabel=L"$\overline{\chi}\"(k,\, \omega)$", 
                   xscale=log10, yscale=log10, 
                   xticklabelsvisible=false,
                #    xminorticks = IntervalsBetween(9),xminorticksvisible=true,xminortickalign=1,yminorticks = IntervalsBetween(9),yminorticksvisible=true,yminortickalign=1,
                   limits=(10.0^-14, 10.0^3, 10.0^-3, 10.0^-0.3), 
                   title=L"$\delta = %$(round(δ_list_to_plot[2], digits=2))$", xgridvisible=false, ygridvisible=false)
                #    , xlabelsize=20, ylabelsize=20)
    
    text!(fig_ax1_top, 2*10.0^-14, 10.0^-0.6, 
          text=L"$\text{(c)}$")
    text!(fig_ax1, 2*10.0^-14, 10.0^-0.6, 
          text=L"$\text{(a)}$")
    text!(fig_ax2_top, 2*10.0^-14, 10.0^-0.6, 
          text=L"$\text{(d)}$")
    text!(fig_ax2, 2*10.0^-14, 10.0^-0.6, 
          text=L"$\text{(b)}$")

    for δ in δ_list_to_plot
        ϕ_arr = Float64[]
        for sol in files_mct_sol
            f = h5open(joinpath(dir, sol), "r")
            δ_sol = std(read(f["run_params"], "D_arr") ; corrected=false)

            if δ_sol == δ

                push!(ϕ_arr, read(f["run_params"], "phi"))

                println("found matching δ file !")

            end
            close(f)
        end
        
        # state closest to the transition
        ϕmax = maximum(ϕ_arr)
        sol_name = "S"
        for sol in files_mct_sol
            if occursin(string(ϕmax), sol)
                sol_name = sol
                break
            end
        end

        colmap = cgrad(:copper, length(ϕ_arr)+1, categorical = true)

        f = h5open(joinpath(dir, sol_name), "r")
        Ns = read(f["run_params"], "Ns")
        t_array = read(f["F_tagged"], "t_array")
        ω_array = read(f["relaxation_spectrum"], "omega_arr")

        Fs_avg = zeros(length(t_array))
        chi_avg = zeros(length(ω_array))

        for s in 1:Ns 
            Fs_avg .+= read(f["F_tagged"], "F_s_$(s)") ./ Ns
            chi_avg .+= read(f["relaxation_spectrum"], "chi_s_$(s)") ./ Ns
        end

        close(f)

        β_KWW_avg1 = 0.0
        β_KWW_avg2 = 0.0
        
        col1 = 2
        col2 = 2
        
        if δ == δ_list_to_plot[1]
            ϕc = find_critical_volume_fraction(δ, dir_crit)
            crit_filename = find_crit_filename(dir_crit, ϕc)

            fc_avg = find_fc_avg(dir_crit, crit_filename)

            if fc_avg > 10.0^-2

                τ_c = find_relaxation_time(t_array, Fs_avg ; threshold = fc_avg/exp(1))
                fmin = fc_avg / 10
                fmax = fc_avg / 2

                β_KWW_avg1 = fit_stretched_exponential(fmin, fmax, t_array, Fs_avg, fc_avg).val
                @show β_KWW_avg1
                χdata_KWW, ω_arr_KWW = compute_relaxation_spectrum_trap(t_array[2:end], fc_avg.*exp.(.-(t_array[2:end]./τ_c).^β_KWW_avg1))

                f = h5open(joinpath(dir, sol_name), "r")
                Ns = read(f["run_params"], "Ns")
                t_array = read(f["F_tagged"], "t_array")
                ω_array = read(f["relaxation_spectrum"], "omega_arr")
                
                chi_avg = zeros(length(ω_array))
        
                for s in 1:Ns 
                    chi_avg .+= read(f["relaxation_spectrum"], "chi_s_$(s)") ./ Ns
                end

                lines!(fig_ax1, ω_arr_KWW, χdata_KWW, 
                       linestyle=:dash, color=colmap[1])

                lines!(fig_ax1_top, ω_arr_KWW, χdata_KWW, 
                       linestyle=:dash, color=colmap[1])

                for s in 1:Ns 
                    chi_s = read(f["relaxation_spectrum"], "chi_s_$(s)")
                    lines!(fig_ax1_top, ω_array, chi_s, color=(colmap[end-1], 0.8))#, linestyle=:dash)
                end

                scatter!(fig_ax1_top, ω_array[1:3:end], chi_avg[1:3:end], color=colmap[1])

                close(f)
            end

            γ_temp = find_γ_exponent(δ)
            γ, err_γ = γ_temp.val, γ_temp.err
            a_temp, b_temp = find_a_b_exponents(γ)
            a, err_a = a_temp.val, a_temp.err
            b, err_b = b_temp.val, b_temp.err

            ω_arr_b = LinRange(10.0^-9, 5*10.0^-8, 10)
            ω_arr_a = LinRange(10.0^-4, 10.0^-2, 10)

            lines!(fig_ax1_top, ω_arr_b, 10.0^-6 .*ω_arr_b.^(-b), 
                   color="black")
            lines!(fig_ax1_top, ω_arr_a, 5*10.0^-2 .*ω_arr_a.^(a),
                   color="black")

            text!(fig_ax1_top, ω_arr_b[end-6], 2*10.0^-2,
                   text=L"$\omega^{-b}$")
             text!(fig_ax1_top, ω_arr_a[end-1], 6*10.0^-3,
                   text=L"$\omega^{\, a}$")
 

            for sol in files_mct_sol
                f = h5open(joinpath(dir, sol), "r")
                δ_loc = std(read(f["run_params"], "D_arr") ; corrected=false)
                ϕ_loc = read(f["run_params"], "phi")
                
                if δ_loc == δ && abs(ϕ_loc-ϕc)<10.0^-2
                    t_array_loc = read(f["F_tagged"], "t_array")
                    ω_array_loc = read(f["relaxation_spectrum"], "omega_arr")
            
                    Fs_avg_loc = zeros(length(t_array))
                    chi_avg_loc = zeros(length(ω_array))
            
                    for s in 1:Ns 
                        Fs_avg_loc .+= read(f["F_tagged"], "F_s_$(s)") ./ Ns
                        chi_avg_loc .+= read(f["relaxation_spectrum"], "chi_s_$(s)") ./ Ns
                    end

                    τ_loc = find_relaxation_time(t_array_loc, Fs_avg_loc ; threshold = fc_avg/exp(1))
                    χdata_KWW_loc, ω_arr_KWW_loc = compute_relaxation_spectrum_trap(t_array[2:end], fc_avg.*exp.(.-(t_array[2:end]./τ_loc).^β_KWW_avg1))

                    scatter!(fig_ax1, ω_array_loc[1:3:end], chi_avg_loc[1:3:end], color=colmap[end-col1])
                    lines!(fig_ax1, ω_arr_KWW_loc, χdata_KWW_loc, 
                           color=colmap[end-col1], linestyle=:dash)
                    col1 += 1
                end
                close(f)
            end

        end 

        if δ == δ_list_to_plot[2]

            ϕc = find_critical_volume_fraction(δ, dir_crit)
            crit_filename = find_crit_filename(dir_crit, ϕc)

            fc_avg = find_fc_avg(dir_crit, crit_filename)

            if fc_avg > 10.0^-2

                τ_c = find_relaxation_time(t_array, Fs_avg ; threshold = fc_avg/exp(1))
                fmin = fc_avg / 10
                fmax = fc_avg / 2

                β_KWW_avg2 = fit_stretched_exponential(fmin, fmax, t_array, Fs_avg, fc_avg).val
                @show β_KWW_avg2

                χdata_KWW, ω_arr_KWW = compute_relaxation_spectrum_trap(t_array[2:end], fc_avg.*exp.(.-(t_array[2:end]./τ_c).^β_KWW_avg2))

                f = h5open(joinpath(dir, sol_name), "r")
                Ns = read(f["run_params"], "Ns")
                t_array = read(f["F_tagged"], "t_array")
                ω_array = read(f["relaxation_spectrum"], "omega_arr")
                
                chi_avg = zeros(length(ω_array))
        
                for s in 1:Ns 
                    chi_avg .+= read(f["relaxation_spectrum"], "chi_s_$(s)") ./ Ns
                end
                
                lines!(fig_ax2, ω_arr_KWW, χdata_KWW, 
                       linestyle=:dash, color=colmap[1])

                lines!(fig_ax2_top, ω_arr_KWW, χdata_KWW, 
                       linestyle=:dash, color=colmap[1])

                for s in 1:Ns 
                    chi_s = read(f["relaxation_spectrum"], "chi_s_$(s)")
                    lines!(fig_ax2_top, ω_array, chi_s, color=(colmap[end-1], 0.8))#, linestyle=:dash)
                end
                
                scatter!(fig_ax2_top, ω_array[1:3:end], chi_avg[1:3:end], color=colmap[1])

                close(f)
            end

            γ_temp = find_γ_exponent(δ)
            γ, err_γ = γ_temp.val, γ_temp.err
            a_temp, b_temp = find_a_b_exponents(γ)
            a, err_a = a_temp.val, a_temp.err
            b, err_b = b_temp.val, b_temp.err

            ω_arr_b = LinRange(10.0^-9, 5*10.0^-8, 10)
            ω_arr_a = LinRange(10.0^-4, 10.0^-2, 10)

            lines!(fig_ax2_top, ω_arr_b, 10.0^-6 .*ω_arr_b.^(-b), 
                   color="black")
            lines!(fig_ax2_top, ω_arr_a, 5*10.0^-2 .*ω_arr_a.^(a),
                   color="black")

            text!(fig_ax2_top, ω_arr_b[end-6], 2*10.0^-2,
                  text=L"$\omega^{-b}$")
            text!(fig_ax2_top, ω_arr_a[end-1], 6*10.0^-3,
                  text=L"$\omega^{\, a}$")

            for sol in files_mct_sol
                f = h5open(joinpath(dir, sol), "r")
                δ_loc = std(read(f["run_params"], "D_arr") ; corrected=false)
                ϕ_loc = read(f["run_params"], "phi")
                if δ_loc == δ && abs(ϕ_loc-ϕc)<10.0^-2
                    t_array_loc = read(f["F_tagged"], "t_array")
                    ω_array_loc = read(f["relaxation_spectrum"], "omega_arr")
            
                    Fs_avg_loc = zeros(length(t_array))
                    chi_avg_loc = zeros(length(ω_array))
            
                    for s in 1:Ns 
                        Fs_avg_loc .+= read(f["F_tagged"], "F_s_$(s)") ./ Ns
                        chi_avg_loc .+= read(f["relaxation_spectrum"], "chi_s_$(s)") ./ Ns
                    end

                    τ_loc = find_relaxation_time(t_array_loc, Fs_avg_loc ; threshold = fc_avg/exp(1))
                    χdata_KWW_loc, ω_arr_KWW_loc = compute_relaxation_spectrum_trap(t_array[2:end], fc_avg.*exp.(.-(t_array[2:end]./τ_loc).^β_KWW_avg2))

                    scatter!(fig_ax2, ω_array_loc[1:3:end], chi_avg_loc[1:3:end], color=colmap[end-col2])
                    lines!(fig_ax2, ω_arr_KWW_loc, χdata_KWW_loc,
                           color=colmap[end-col2], linestyle=:dash)
                    col2 += 1
                end
                close(f)
            end

        end

    end
    text!(fig_ax1, 10.0^-8, 2.5*10.0^-1, 
          text=L"$\varphi_c\longleftarrow\varphi$")
    text!(fig_ax2, 10.0^-8, 2*10.0^-1, 
          text=L"$\varphi_c\longleftarrow\varphi$")
    rowgap!(fig.layout, 0)

    #text!(fig_ax1_top, 10.0^-14, 10.0^-2,
    #     text=L"$\longleftarrow$", fontsize=35)
    #text!(fig_ax1_top, 10.0^-6, 1.5*10.0^-3,
    #      text=L"$\downarrow$", fontsize=45)

    # invisible axis spanning the whole figure, used to draw the "D" arrows
    axall = Axis(fig[:, :], limits=(0,1,0,1))
    hidedecorations!(axall)
    hidespines!(axall)

    arrows!(axall, [0.075], [0.27],
                   [-0.06], [0.1], 
                   color="black")
    text!(axall, 0.075, 0.23, text=L"$D$")

    arrows!(axall, [0.2], [0.12],
                   [-0.04], [-0.06], 
    color="black")

    text!(axall, 0.13, 0.07, text=L"$D$")

    save("Plots/susceptibility_epsilon.pdf", fig)
    display(fig)
end

# plot_scaled_susceptibility_ϵ()

"""
    plot_TTS()

Time-temperature superposition test: for each δ, plot F̄_s vs t/τ_α for all volume fractions,
with τ_α defined by F̄_s(τ_α) = 10⁻². One figure per δ.
"""
function plot_TTS()
    dir = "Data_sol_uniform_epsilon_paper"
    files_mct_sol = String[]

    for file in readdir(dir)
        if occursin("mctsol", file)
            push!(files_mct_sol, file)
        end
    end

    δ_list = Any[]
    for file_sol in files_mct_sol
        f = h5open(joinpath(dir, file_sol), "r")
        δ = std(read(f["run_params"], "D_arr") ; corrected=false)
        Ns = read(f["run_params"], "Ns")

        close(f)
        if Ns == 10
            push!(δ_list, δ)
        end
    end

    δ_list = sort(union(δ_list))

    for δ in δ_list
        fontsize_theme = Theme(fontsize=25)
        set_theme!(fontsize_theme)
        Makie.theme(:fonts).:regular = "CMU Serif"
    
        fig = Figure(resolution=(800,600))
        ax = Axis(fig[1,1], xscale=log10, limits=(10.0^-5,100,0,1), 
                  xlabel=L"$t/\overline{\tau}_{\alpha}$", ylabel=L"$\overline{F}(k,t/\overline{\tau}_{\alpha})$", 
                  title=L"\delta=%$(round(δ, digits=3))")
        for sol in files_mct_sol
            f = h5open(joinpath(dir, sol), "r")
            Ns = read(f["run_params"], "Ns")
            t_array = read(f["F_tagged"], "t_array")[2:end]
            Fs_avg = zeros(length(t_array))
            ϕ = read(f["run_params"], "phi")
            if std(read(f["run_params"], "D_arr") ; corrected=false) == δ
                for s in 1:Ns
                    Fs_avg .+= read(f["F_tagged"], "F_s_$(s)")[2:end] ./Ns
                end

                τ_α = find_relaxation_time(t_array, Fs_avg; threshold=10.0^-2)

                scatter!(ax, t_array[1:4:end]./τ_α, Fs_avg[1:4:end], 
                         label=L"$\phi=%$(ϕ)$")

            else
                continue
            end

            close(f)
        end

        axislegend(ax, position = :rt)

        display(fig)
    end

end

# plot_TTS()


"""
    plot_ISF()

Plot F̄_s(t) for every MCT solution of the inverse-cubic distribution (`Data_sol_Inverse_Cubic_epsilon`).
"""
function plot_ISF()
    dir = "Data_sol_Inverse_Cubic_epsilon"

    files_mct_sol = String[]
    
    for file in readdir(dir)
        if occursin("mctsol", file)
            push!(files_mct_sol, file)
        end
    end
    
    fig = Figure(resolution=(500,500))

    fontsize_theme = Theme(fontsize=20)
    set_theme!(fontsize_theme)
    Makie.theme(:fonts).:regular = "CMU Serif"

    ax = Axis(fig[1, 1], xtickalign=1, ytickalign=1, ylabel=L"$\overline{F_s}(k,\, t)$", 
               xlabel=L"$t$", xscale=log10, 
            #    xminorticks = IntervalsBetween(9),xminorticksvisible=true,xminortickalign=1,
               limits=(10.0^-3, 10.0^11, 0, 1))

    for fname in files_mct_sol
        f = h5open(joinpath(dir, fname), "r")
        t_arr = read(f["F_tagged"], "t_array")
        Fs_avg = zeros(length(t_arr))
        Ns = read(f["run_params"], "Ns")
        
        for i in 1:Ns
            Fs_avg .+= read(f["F_tagged"], "F_s_$(i)") ./ Ns
        end

        lines!(ax,t_arr[1:end-1], Fs_avg[1:end-1])
        close(f)    
    end
    display(fig)
end

# plot_ISF()

