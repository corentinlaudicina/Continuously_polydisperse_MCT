#=
Fit Kohlrausch-Williams-Watts (KWW) stretched exponentials F(t) = f_c exp(-(t/τ_α)^β) to the
α-relaxation of the tagged correlation functions, and compare their susceptibilities with the
MCT ones.

When run, the script takes the inverse-cubic data (`Data_sol_Inverse_Cubic_epsilon`, critical
points in `Data_Phase_Diagram_Inverse_Cubic`) at a fixed δ. It fits β_KWW to the species-averaged
F̄_s at the state point closest to the transition (largest ϕ), then plots F̄_s(t) and χ̄''(ω) at
every ϕ together with the KWW curves for that β (and each state's own τ_α).
=#

using ModeCouplingTheory
using Dierckx, QuadGK
using LsqFit, Plots, SpecialFunctions
using LaTeXStrings, StaticArrays
using DelimitedFiles, Statistics
using HDF5

include("find_relaxation_time.jl")

"""
    find_critical_volume_fraction(δ, datafolder_crit)

Return the smallest volume fraction with a glass solution (sum of the collective non-ergodicity
parameter > 10⁻⁴), from the files `..._poly_<δ>_phi_<ϕ>.jld2` in `datafolder_crit`.
"""
function find_critical_volume_fraction(δ, datafolder_crit)

    files = readdir(datafolder_crit)
    println("$(length(files)) files found")
    files = files[contains.(files, "poly_$(δ)_")]
    mydict = Dict{Float64, Float64}()

    for file in files
        jldopen(joinpath(datafolder_crit, file)) do f
            fc = f["non_erg_param"]["fc_collective"]
            ϕ = parse(Float64, split(file, '_')[7][1:end-5])
            mydict[ϕ] = sum(sum(fc))
        end
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
    fit_stretched_exponential(fmin, fmax, t_arr, F, Fs_c)

Fit the KWW exponent β of `F(t) ≈ Fs_c exp(-(t/τ_α)^β)` and return it.

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
    fit_params = [0.1]
    while count < maxcount

        fit = curve_fit(powerlaw, t_to_fit, log10.(F_to_fit), p0)
        fit_params[1] = fit.param[1]
        @show se = sqrt.(estimate_covar(fit))
        if abs(fit_params[1]-p0[1])/p0[1] > 10.0^-4
            break 
        else 
            p0[1] = rand() 
        end

        count +=1
    end
    println("RETURNING KWW EXPONENT, $(count) ITERATIONS REQUIRED")
    return fit_params[1]
end

"""
    find_corresponding_sol(list_mct_sol, Ns, Δ)

Return the file names in `list_mct_sol` (`mctsol_Ns_<Ns>_Delta_<Δ>_...`) with the given `Ns` and `Δ`.
"""
function find_corresponding_sol(list_mct_sol, Ns, Δ)
    list_mct_sol_same_params = String[]
    for sol in list_mct_sol
        name = split(sol, '_')
        sol_Ns = parse(Int64, name[3])
        sol_Δ = parse(Float64, name[5])

        if sol_Ns == Ns && sol_Δ == Δ
            push!(list_mct_sol_same_params, sol)
        end
    end
    println("There are $(length(list_mct_sol_same_params)) files with parameters Ns=$(Ns), Δ=$(Δ)")
    return list_mct_sol_same_params
end

"""
    get_params(dir)

Read the run parameters stored in `dir`: Δ values from `Delta_array.txt`, ϵ values from
`epsilon_array.txt`, and the numbers of species from the `phi_c_..._<Ns>.txt` file names.
Returns `(Δ_arr, ϵ_arr, Ns_arr)`.
"""
function get_params(dir)

    file_Δ = "Delta_array.txt"
    file_ϵ = "epsilon_array.txt"

    Δ_arr = readdlm(joinpath(dir,file_Δ), '\t', Float64, '\n')#[1, 1:end]
    ϵ_arr = readdlm(joinpath(dir,file_ϵ), '\t', Float64, '\n')#[1, 1:end]
    Ns_arr = []
    for file in readdir(dir)
        if occursin("phi_c", file)
            Ns = parse(Int64, split(file, '_')[4][1:end-4])
            push!(Ns_arr, Ns)
        end
    end
    Ns_arr = sort(union(Ns_arr))
    return Δ_arr, ϵ_arr, Ns_arr
end

"""
    calculate_avg_Fs_and_avg_χ(dir_MCT_sol, sol_filename)

Read an MCT solution file and return the species averages `(F̄_s, t, χ̄'', ω)`.
"""
function calculate_avg_Fs_and_avg_χ(dir_MCT_sol, sol_filename)

    f = jldopen(joinpath(dir_MCT_sol, sol_filename), "r")

    Ns = f["run_params"]["Ns"]
    t_array = f["F_tagged"]["t_array"]
    ω_array = f["relaxation_spectrum"]["omega_arr"]
    
    Fs_avg = zeros(length(t_array))
    χs_avg = zeros(length(ω_array))

    for s in 1:Ns 
        Fs_avg .+= f["F_tagged"]["F_s_$(s)"] ./ Ns
        χs_avg .+= f["relaxation_spectrum"]["chi_s_$(s)"] ./ Ns
    end

    close(f)
    return Fs_avg, t_array, χs_avg, ω_array
end

"""
    find_k_peak(Ns, δ, dir_MCT_sol)

Return the wave vector `k_peak` stored in the solution file with `Ns` species and diameter
standard deviation `δ`, or 0 if it is not found.

Note: only the first file of `dir_MCT_sol` is checked, since both branches of the loop return.
"""
function find_k_peak(Ns, δ, dir_MCT_sol)
    for file in readdir(dir_MCT_sol)
        f = jldopen(joinpath(dir_MCT_sol, file), "r")
        Ns_file = f["run_params"]["Ns"]
        δ_file = std(f["run_params"]["D_arr"] ; corrected=false)
        close(f)

        if Ns == Ns_file && δ == δ_file
            f = jldopen(joinpath(dir_MCT_sol, file), "r")
            k_peak = f["run_params"]["k_peak"]
            close(f)
            return k_peak
        else
            println("file not found, returning 0 for k_peak")
            return 0
        end
    end
end

"""
    find_k_peak_id(Ns, δ, dir_MCT_sol)

Return the index of [`find_k_peak`](@ref) in the grid `range(0.2, 39.8, length=100)` (0 if absent).
"""
function find_k_peak_id(Ns, δ, dir_MCT_sol)
    k_array = range(0.2, 39.8, length=100)

    k_peak = find_k_peak(Ns, δ, dir_MCT_sol)
    k_peak_id = 0 

    for kid in 1:length(k_array)
        if k_array[kid] == k_peak
            k_peak_id = kid
            break 
        end
    end

    return k_peak_id
end

"""
    find_critical_plateau(ϕc_hi, s, k_id, dir_MCT_crit)

Return the tagged non-ergodicity parameter f_c^(s)(k) of species `s` at wave-vector index `k_id`,
from the critical-point file for `ϕc_hi` in `dir_MCT_crit`.
"""
function find_critical_plateau(ϕc_hi, s, k_id, dir_MCT_crit)

    for file in readdir(dir_MCT_crit)
        if occursin("phi_$(ϕc_hi)", file)
            f = jldopen(joinpath(dir_MCT_crit, file), "r")
            fc_s = f["non_erg_param"]["f_c_$(s)"]
            close(f)
            return fc_s[k_id]
        else
            continue
        end
    end
    println("WARNING ! NEP file for ϕ_c = $(ϕc_hi) not found in "*dir_MCT_crit)
end

"""
    fit_leading_critical_eigvec(Fs_avg, t_array, fc_s, a, b)

Fit the leading-order β-relaxation laws around the plateau `fc_s`: the critical decay
`fc_s + A t^(-a)` on the points with `fc_s < F < 1.05 fc_s`, and the von Schweidler law
`fc_s - B t^b` on the points with `0.95 fc_s < F < fc_s`. Returns `A`, and warns if `A` and `B`
differ by more than 10%.
"""
function fit_leading_critical_eigvec(Fs_avg, t_array, fc_s, a, b)

    indices_to_fit_a = [i for i in eachindex(Fs_avg) if fc_s < Fs_avg[i] < 1.05*fc_s]
    indices_to_fit_b = [i for i in eachindex(Fs_avg) if 0.95*fc_s < Fs_avg[i] < fc_s]

    leading_order_a(t, p) = fc_s .+ p[1].*t.^(-a)
    leading_order_b(t, p) = fc_s .- p[1].*t.^(b)

    p0_a = [rand()]
    p0_b = [rand()]

    fit_a = curve_fit(leading_order_a, t_array[indices_to_fit_a], Fs_avg[indices_to_fit_a], p0_a)
    fit_b = curve_fit(leading_order_b, t_array[indices_to_fit_b], Fs_avg[indices_to_fit_b], p0_b)

    crit_amplitude_a = fit_a.param[1]
    crit_amplitude_b = fit_b.param[1]
    
    rel_err = abs(crit_amplitude_a - crit_amplitude_b)/crit_amplitude_b
    if rel_err > 0.1
        println("WARNING ! Large error in critical amplitude ! rel_err = $(rel_err)")
    end

    return crit_amplitude_a
end

"""
    fit_next_leading_order_crit_eigenvec(Fs_avg, t_array, fc_s, a, b, crit_amplitude_a)

Fit the next-to-leading-order amplitudes `A₂` and `B₂` of `fc_s + A t^(-a) + A₂ t^(-2a)` and
`fc_s - A t^b - B₂ t^(2b)`, with the leading amplitude `A = crit_amplitude_a` fixed, on the points
within 10% of the plateau. Returns `A₂`, and warns if `A₂` and `B₂` differ by more than 10%.
"""
function fit_next_leading_order_crit_eigenvec(Fs_avg, t_array, fc_s, a, b, crit_amplitude_a)

    next_leading_order_a(t, p) = fc_s .+ crit_amplitude_a.*t.^(-a) .+ p[1].*t.^(-2*a)
    next_leading_order_b(t, p) = fc_s .- crit_amplitude_a.*t.^(b) .- p[1].*t.^(2*b)

    indices_to_fit_a_next = [i for i in eachindex(Fs_avg) if fc_s < Fs_avg[i] < 1.1*fc_s]
    indices_to_fit_b_next = [i for i in eachindex(Fs_avg) if 0.9*fc_s < Fs_avg[i] < fc_s]


    p0_a_next = [rand()]
    p0_b_next = [rand()]

    fit_a_next = curve_fit(next_leading_order_a, t_array[indices_to_fit_a_next], Fs_avg[indices_to_fit_a_next], p0_a_next)
    fit_b_next = curve_fit(next_leading_order_b, t_array[indices_to_fit_b_next], Fs_avg[indices_to_fit_b_next], p0_b_next)

    crit_amplitude_a_next = fit_a_next.param[1]
    crit_amplitude_b_next = fit_b_next.param[1]

    rel_err = abs(crit_amplitude_a_next - crit_amplitude_b_next)/crit_amplitude_b_next
    if rel_err > 0.1
        println("WARNING ! Large error in critical amplitude !, rel_err = $(rel_err)")
    end
    return crit_amplitude_a_next
end

"""
    write_stretched_exponents(dir_MCT_crit, dir_MCT_sol, sol_filename)

Fit β_KWW for every species with plateau f_c^(s) > 0.1 and for the species average, using the
window `f_c/10 < F < f_c/2`. Intended to store the results in the group `KWW_exponents` of the
solution file, but the writes are commented out.
"""
function write_stretched_exponents(dir_MCT_crit, dir_MCT_sol, sol_filename)

    f = h5open(joinpath(dir_MCT_sol, sol_filename), "r+")

    Ns = read(f["run_params"], "Ns")
    δ = std(read(f["run_params"], "D_arr") ; corrected=false)
    close(f)

    Fs_avg, t_array, χs_avg, ω_array = calculate_avg_Fs_and_avg_χ(dir_MCT_sol, sol_filename)
    
    ϕc_hi = find_critical_volume_fraction(δ, Ns, dir_MCT_crit)
    
    fc_avg = 0.0

    k_id = find_k_peak_id(Ns, δ, dir_MCT_sol)

    if "KWW_exponents" keys(f)
        println("KWW_exponents already determined, returning nothing.")
        return 
    else
        # create_group(f, "KWW_exponents")
    end

    for s in 1:Ns
        Fs = f["F_tagged"]["F_s_$(s)"]

        fc_s = find_critical_plateau(ϕc_hi, s, k_id, dir_MCT_crit)
        fmin_s = fc_s/10
        fmax_s = fc_s/2

        if fc_s > 10.0^-1
            β_KWW_s = fit_stretched_exponential(fmin_s, fmax_s, t_array, Fs, fc_s)
            # write(f["KWW_exponents"], "beta_KWW_s_$(s)", β_KWW_s)
        else
            # write(f["KWW_exponents"], "beta_KWW_s_$(s)", 0.0)
        end

        fc_avg += fc_s./ Ns
    end

    fmin = fc_avg / 10
    fmax = fc_avg / 2
    β_KWW_avg = fit_stretched_exponential(fmin, fmax, t_array, Fs_avg, fc_avg)

    # write(f["KWW_exponents"], "beta_KWW_avg", β_KWW_avg)
    close(f)
    println("KWW exponents written.")
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

# Script: KWW fits for the inverse-cubic distribution at fixed δ
dir_MCT_sol = "Data_sol_Inverse_Cubic_epsilon"
dir_MCT_crit = "Data_Phase_Diagram_Inverse_Cubic"
crit_sol_list = readdir(dir_MCT_crit)
sol_list = readdir(dir_MCT_sol)
δ = 0.22537088908436007

# plateau f̄_c from the critical-point file
ϕhi = find_critical_volume_fraction(δ, dir_MCT_crit)
file_to_fit = "a"
for file in readdir(dir_MCT_crit)
    if occursin(string(ϕhi), file)
        file_to_fit = file
    end
end

fc_avg = find_fc_avg(dir_MCT_crit, file_to_fit)

# state point closest to the transition
ϕ_arr = []
for sol in sol_list
    f = h5open(joinpath(dir_MCT_sol, sol), "r")
    ϕ = read(f["run_params"] ,"phi")
    push!(ϕ_arr, ϕ)
    close(f)
end

ϕmax = maximum(ϕ_arr)
p = plot(xscale=:log, xlims=(10.0^-3, 10.0^14), ylims=(0,1))
pX = plot(xscale=:log, yscale=:log)
# fit β_KWW at ϕmax, and plot F̄_s(t) and χ̄''(ω) with the KWW curves
β_KWW_avg = 0.0
for sol in sol_list
    if occursin(string(ϕmax), sol)
        Fs_avg, t_array, χs_avg, ω_array = calculate_avg_Fs_and_avg_χ(dir_MCT_sol, sol)
        
        fmin = fc_avg / 10
        fmax = fc_avg / 2
        β_KWW_avg += fit_stretched_exponential(fmin, fmax, t_array, Fs_avg, fc_avg)
        τ_α = find_relaxation_time(t_array,Fs_avg; threshold=fc_avg/exp(1))

        scatter!(p, t_array[2:10:end], Fs_avg[2:10:end], markerstrokewidth=0.0)
        plot!(p, t_array[2:end], fc_avg.*exp.(.-(t_array[2:end]./τ_α).^β_KWW_avg))

        Fs_avg, t_array, χs_avg, ω_array = calculate_avg_Fs_and_avg_χ(dir_MCT_sol, sol)
        scatter!(pX, ω_array[1:2:end], χs_avg[1:2:end], markerstrokewidth=0.0)


        χdata_KWW, ω_arr_KWW = compute_relaxation_spectrum(t_array[2:end], fc_avg.*exp.(.-(t_array[2:end]./τ_α).^β_KWW_avg))

        plot!(pX, ω_arr_KWW, χdata_KWW)
        end
end

# other state points: KWW curves with the β fitted at ϕmax and their own τ_α
for sol in sol_list
    if occursin(string(ϕmax), sol) == false
        Fs_avg, t_array, χs_avg, ω_array = calculate_avg_Fs_and_avg_χ(dir_MCT_sol, sol)
        
        τ_α = find_relaxation_time(t_array,Fs_avg; threshold=fc_avg/exp(1))

        scatter!(p, t_array[2:10:end], Fs_avg[2:10:end], markerstrokewidth=0.0)
        plot!(p, t_array[2:end], fc_avg.*exp.(.-(t_array[2:end]./τ_α).^β_KWW_avg))

        Fs_avg, t_array, χs_avg, ω_array = calculate_avg_Fs_and_avg_χ(dir_MCT_sol, sol)
        scatter!(pX, ω_array[1:2:end], χs_avg[1:2:end], markerstrokewidth=0.0)


        χdata_KWW, ω_arr_KWW = compute_relaxation_spectrum(t_array[2:end], fc_avg.*exp.(.-(t_array[2:end]./τ_α).^β_KWW_avg))

        plot!(pX, ω_arr_KWW, χdata_KWW)

    end
end

# Older analyses (per-file KWW and critical-law fits), kept for reference:
# hline!(p, [fc_avg])

# for sol in readdir(dir_MCT_sol)
#     f = h5open(joinpath(dir_MCT_sol, sol), "r")

#     t_array = read(f["F_tagged"], "t_array")
#     Fs_avg = zeros(length(t_array))
#     Ns = read(f["run_params"], "Ns")

#     for s in 1:Ns 
#         Fs_avg .+= read(f["F_tagged"], "F_s_$(s)") ./ Ns 
#     end

#     scatter!(p, t_array[2:10:end], Fs_avg[2:10:end])
#     close(f)
# end
# display(p)


# f = jldopen(joinpath(dir_MCT_crit, ))

# sol_list = readdir(dir_MCT_sol)
# for sol_filename in sol_list
#     write_stretched_exponents(dir_MCT_crit, dir_MCT_sol, sol_filename)
# end
# sol_files = readdir(dir_MCT_sol)

# p = plot(xscale=:log, xlims=(10.0^-3, 10.0^14), ylims=(0,1))
# pX = plot(xscale=:log, yscale=:log)

# sol_files = sort!(sol_files)
# for sol in sol_files

    # p = plot(xscale=:log, xlims=(10.0^-3, 10.0^14), ylims=(0,1))
    # pX = plot(xscale=:log, yscale=:log)

    # f = h5open(joinpath(dir_MCT_sol, sol), "r")

    # Ns = read(f["run_params"], "Ns")
    # δ = std(read(f["run_params"], "D_arr") ; corrected=false)
    # close(f)
    # k_id = find_k_peak_id(Ns, δ, dir_MCT_sol)

    # Fs_avg, t_array, χs_avg, ω_array = calculate_avg_Fs_and_avg_χ(Ns, δ, dir_MCT_sol)
    # ϕc_hi = find_critical_volume_fraction(δ, Ns, dir_MCT_crit)
    # fc_s = find_critical_plateau(Ns, δ ,ϕc_hi, s, k_id, dir_MCT_crit)

    # f = jldopen(joinpath(dir_MCT_sol, sol), "r")

    # t_array = f["F_tagged"]["t_array"]
    # ω_array = f["relaxation_spectrum"]["omega_arr"]
    # Ns = f["run_params"]["Ns"]
    # δ = std(f["run_params"]["D_arr"] ; corrected=false)
    # Fs_avg = zeros(length(t_array))
    # χs_avg = zeros(length(ω_array))

    # if δ == 0 
    #     continue 
    # end
    # for s in 1:Ns
    #     Fs_avg .+= f["F_tagged"]["F_s_$(s)"] ./ Ns
    #     χs_avg .+= f["relaxation_spectrum"]["chi_s_$(s)"] ./ Ns
    # end
    # close(f)

    # τ_α = find_relaxation_time(t_array, Fs_avg ; threshold=fc_s/exp(1))

    # fmax = fc_s / 2
    # fmin = fc_s / 10
    
    # β_KWW = fit_stretched_exponential(fmin, fmax, t_array, Fs_avg, fc_s)

    # Fs_avg_Kolsrauch = fc_s.*exp.(.-(t_array ./τ_α).^β_KWW)

    # χs_avg_Kolsrauch, ω_arr_Kolsrauch = compute_relaxation_spectrum(t_array[2:end], Fs_avg_Kolsrauch[2:end])

    # scatter!(p, t_array[2:10:end], Fs_avg[2:10:end], label=false, markerstrokewidth =0.0)

    # plot!(p, t_array[2:end], Fs_avg_Kolsrauch[2:end], label=false)

    # fit_a = curve_fit(leading_order_a, t_array[indices_to_fit_a], Fs_avg[indices_to_fit_a], p0_a)
    # fit_b = curve_fit(leading_order_b, t_array[indices_to_fit_b], Fs_avg[indices_to_fit_b], p0_b)

    # crit_amplitude_a = [fit_a.param[1]]
    # crit_amplitude_b = [fit_b.param[1]]

    # next_leading_order_a(t, p) = fc_s .+ crit_amplitude_a[1].*t.^(-a) .+ p[1].*t.^(-2*a)
    # next_leading_order_b(t, p) = fc_s .- crit_amplitude_b[1].*t.^(b) .- p[1].*t.^(2*b)

    # p0_a_next = [rand()]
    # p0_b_next = [rand()]

    # fit_a_next = curve_fit(next_leading_order_a, t_array[indices_to_fit_a_next], Fs_avg[indices_to_fit_a_next], p0_a_next)
    # fit_b_next = curve_fit(next_leading_order_b, t_array[indices_to_fit_b_next], Fs_avg[indices_to_fit_b_next], p0_b_next)

    # crit_amplitude_a_next = [fit_a_next.param[1]]
    # crit_amplitude_b_next = [fit_b_next.param[1]]

    # @show crit_amplitude_a_next, crit_amplitude_b_next

    # plot!(p, t_array[2:end], leading_order_a.(t_array[2:end], crit_amplitude_a), ls=:dash, label="leading a")
    # plot!(p, t_array[2:end], leading_order_b.(t_array[2:end], crit_amplitude_b), ls=:dash, label="leading b")

    # plot!(p, t_array[2:end], next_leading_order_a.(t_array[2:end], crit_amplitude_a_next), ls=:dash, label="next leading a")
    # plot!(p, t_array[2:end], next_leading_order_b.(t_array[2:end], crit_amplitude_b_next), ls=:dash, label="next leading b")

    # plot!(p, legend=:bottomleft, title="$(δ)")
    # savefig(p, "zoom.pdf")
    # display(p)

    # χs_avg_a, ω_arr_a = compute_relaxation_spectrum(t_array[2:end], leading_order_a.(t_array[2:end], crit_amplitude_a))
    # χs_avg_b, ω_arr_b = compute_relaxation_spectrum(t_array[2:end], leading_order_b.(t_array[2:end], crit_amplitude_b))

    # χs_avg_a_next, ω_arr_a_next = compute_relaxation_spectrum(t_array[2:end], next_leading_order_a.(t_array[2:end], crit_amplitude_a_next))
    # χs_avg_b_next, ω_arr_b_next = compute_relaxation_spectrum(t_array[2:end], next_leading_order_b.(t_array[2:end], crit_amplitude_b_next))

    # pX = plot(xscale=:log, yscale=:log)
    # scatter!(pX, ω_array[1:2:end], χs_avg[1:2:end], markerstrokewidth =0.0)
    # plot!(pX, ω_arr_Kolsrauch, χs_avg_Kolsrauch./maximum(χs_avg_Kolsrauch), 
    # linestyle=:dash, label="β=$(round(β_KWW, digits=3))")

    # plot!(pX, ω_arr_a, χs_avg_a, linestyle=:dash, label="leading a")
    # plot!(pX, ω_arr_b, χs_avg_b, linestyle=:dash, label="leading a")

    # plot!(pX, ω_arr_a_next, χs_avg_a_next, linestyle=:dash, label="next leading a")
    # plot!(pX, ω_arr_b_next, χs_avg_b_next, linestyle=:dash, label="next leading a")


    # plot!(pX, xlims=(10.0^-14, 10^3), ylims=(10.0^-4, 10.0), title="$(δ)")

    # savefig(pX, "zoomX.pdf")

    # display(pX)
    # plot!(p, title="δ=$(δ)")
    
# end
# plot!(pX, xlims=(10.0^-9, 10^-5), ylims=(10.0^-1, 1.0))
# display(pX)

# display(p)

