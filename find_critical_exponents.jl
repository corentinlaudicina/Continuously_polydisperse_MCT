#=
Extract the MCT exponents γ, a and b for each polydispersity δ (n = 5 species, uniform
distribution, data in `Data_sol_uniform_epsilon`) and compare the data with the asymptotic
MCT laws.

For each δ:
  1. fit τ_α ∝ ε^(-γ) over all distances ε to the critical point, with τ_α defined by
     F̄_s(τ_α) = f̄_c/e;
  2. solve the exponent relation Γ(1-a)²/Γ(1-2a) = Γ(1+b)²/Γ(1+2b), with
     γ = 1/(2a) + 1/(2b), for b, then a;
  3. at the smallest ε, fit the amplitudes of the critical decay F = f_c + A t^(-a) and of the
     von Schweidler law F = f_c - B t^b, and plot them with the data and the KWW fit;
  4. plot the susceptibility χ̄''(ω) of the data, of the KWW fit and of the two power laws.

Reads an older HDF5 layout (`t_array` at the root, `F_tagged/F_s_avg`, `F_tagged_crit`,
`KWW_EXP`, `relaxation_spectrum/chi_avg`) that calculate_MCT_solution.jl no longer writes.
=#

using ModeCouplingTheory
using Dierckx, QuadGK, Plots
using LsqFit, Roots, SpecialFunctions
using LaTeXStrings, StaticArrays
using DelimitedFiles
using HDF5

include("find_relaxation_time.jl")

dir = "Data_sol_uniform_epsilon"

"""
    get_params(mct_sol_files)

Parse the file names `mctsol_Ns_<Ns>_Delta_<δ>_eps_<ϵ>.hdf5` and return the sorted, unique
values `(δ_arr, ϵ_arr, Ns_arr)`.
"""
function get_params(mct_sol_files)

    Ns_arr = []
    δ_arr = []
    ϵ_arr = []
    for file in mct_sol_files
        Ns = parse(Int64, split(file, '_')[3])
        δ = parse(Float64, split(file, '_')[5])
        ϵ = parse(Float64, split(file, '_')[7][1:end-5])

        push!(Ns_arr, Ns)
        push!(δ_arr, δ)
        push!(ϵ_arr, ϵ)

    end
    Ns_arr = sort(union(Ns_arr))
    δ_arr = sort(union(δ_arr))
    ϵ_arr = sort(union(ϵ_arr))

    return δ_arr, ϵ_arr, Ns_arr
end

# MCT solution files, and the subset closest to the transition (smallest ϵ)
files = readdir(dir)
mct_sol_files = []
mct_sol_files_ϵ = []

for file in files 
    if contains(file, "mctsol") # REMEMBER TO CHANGE ! 
        push!(mct_sol_files, file)
    end
end

mct_sol_files = sort(mct_sol_files)
δ_arr, ϵ_arr, Ns_arr = get_params(mct_sol_files)

mct_sol_files_ϵ = []
for file in mct_sol_files 
    if contains(file, string(minimum(ϵ_arr))) # REMEMBER TO CHANGE ! 
        push!(mct_sol_files_ϵ, file)
    end
end

for δ in δ_arr
    # 1. relaxation time τ_α(ϵ) of the species-averaged F_s (Ns = 5 only)
    τ_arr = zeros(length(ϵ_arr))
    ϵ_arr_to_plot = zeros(length(ϵ_arr))
    # p = plot(xscale=:log, yscale=:log)

    count = 1
    for sol_file in mct_sol_files

        f = h5open(joinpath(dir, sol_file), "r")
        ϵ = parse(Float64, split(sol_file, '_')[7][1:end-5])
        Ns = parse(Int64, split(sol_file, '_')[3])
        if contains(sol_file, string(δ)) && Ns == 5
            
            Fs_avg = read(f["F_tagged"],"F_s_avg")
            time_arr = read(f, "t_array")
            F_crit_avg = 0.0

            for s in 1:Ns 
                F_crit_avg += read(f["F_tagged_crit"], "F_s_c_$(s)")
            end

            F_crit_avg /= Ns

            τ_α = find_relaxation_time(time_arr,Fs_avg; threshold=F_crit_avg/exp(1))
            
            τ_arr[count] = τ_α
            ϵ_arr_to_plot[count] = ϵ
            # scatter!(p, [ϵ], [τ_α])
            count += 1
        end
        close(f)
    end

    # fit log10 τ_α = log10 A - γ log10 ϵ
    powerlaw(t, p) = log10(abs(p[1])) .- p[2].*log10.(t)
    p0 = [rand(), rand()]
    fit_γ = curve_fit(powerlaw, ϵ_arr_to_plot, log10.(τ_arr), p0)
    fit_params_γ = [fit_γ.param[1], fit_γ.param[2]]

    γ = fit_params_γ[2]
    @show γ

    # 2. exponent relation Γ(1-a)²/Γ(1-2a) = Γ(1+b)²/Γ(1+2b), with a = 1/(2(γ - 1/(2b)))
    to_solve(b) = gamma(1-1/(2*(γ-1/(2*b))))^2/gamma(1-1/(γ-1/(2*b))) - gamma(1+b)^2/gamma(1+2*b)

    b = find_zero(to_solve, (0.1, 0.8))

    @show b 

    a = 1/(2*(γ-1/(2*b)))

    @show a 

    val_a = gamma(1-a)^2/gamma(1-2*a)
    val_b = gamma(1+b)^2/gamma(1+2*b)

    @show val_a, val_b
    @show rel_err = abs(val_a-val_b)/val_b

    # 3.-4. compare the state closest to ϕ_c with the KWW fit and the asymptotic power laws
    for filename in mct_sol_files_ϵ
        if contains(filename, "Ns_5") && contains(filename, string(δ))
            f = h5open(joinpath(dir, filename), "r")
            t_arr = read(f, "t_array")
            Fs_avg = read(f["F_tagged"], "F_s_avg")

            F_crit_avg = 0.0

            for s in 1:Ns 
                F_crit_avg += read(f["F_tagged_crit"], "F_s_c_$(s)")
            end

            F_crit_avg /= Ns

            β_SB = read(f["KWW_EXP"], "beta_SB")
            τ_α = find_relaxation_time(t_arr,Fs_avg; threshold=F_crit_avg/exp(1))
    
            F_avg_stretched_exp_SB = F_crit_avg .* exp.(.-(t_arr./τ_α).^β_SB)


            p = plot(xscale=:log10)

            scatter!(p, t_arr[2:10:end], Fs_avg[2:10:end], 
                    markerstrokewidth = 0.0, label=false)

            plot!(p, t_arr[2:end], F_avg_stretched_exp_SB[2:end], ls=:dash, label="Kolsrauch")


            # plot!(p, t_arr[2:end], 0.1.*t_arr[2:end].^(-a))      
            # plot!(p, t_arr[2:end], 0.000001 .* t_arr[2:end].^(b))   
            
            # time windows for the critical decay (a) and von Schweidler (b) fits
            tmin_a = 5*10.0^2
            tmax_a = 5*10.0^3

            tmin_b = 5*10.0^8
            tmax_b = 5*10.0^9

            indices_to_fit_a = [i for i in eachindex(t_arr) if tmin_a < t_arr[i] < tmax_a]
            indices_to_fit_b = [i for i in eachindex(t_arr) if tmin_b < t_arr[i] < tmax_b]

            powerlaw_a(t,p) = F_crit_avg .+ p[1].*t.^(-a)
            powerlaw_b(t,p) = F_crit_avg .- p[1].*t.^(b)

            to_fit_a = Fs_avg[indices_to_fit_a]
            t_to_fit_a = t_arr[indices_to_fit_a]
            to_fit_b = Fs_avg[indices_to_fit_b]
            t_to_fit_b = t_arr[indices_to_fit_b]

            p0_a = [rand()]
            p0_b = [rand()]

            fit_a = curve_fit(powerlaw_a, t_to_fit_a, to_fit_a, p0_a)
            fit_params_a = [fit_a.param[1]]

            fit_b = curve_fit(powerlaw_b, t_to_fit_b, to_fit_b, p0_b)
            fit_params_b = [fit_b.param[1]]

            plot!(p, t_arr, powerlaw_a.(t_arr,fit_params_a), label="fc+At^-a")
            plot!(p, t_arr, powerlaw_b.(t_arr,fit_params_b), label="fc-Bt^b")

            # scatter!(p, t_arr[indices_to_fit_a], Fs_avg[indices_to_fit_a])
            # scatter!(p, t_arr[indices_to_fit_b], Fs_avg[indices_to_fit_b])

            plot!(p, xlims=(10.0^-3, 10.0^15), ylims=(0.0, 1.0), legend=:bottomleft)

            display(p)

            # susceptibilities of the data, of the KWW fit and of the two power laws
            χ_avg = read(f["relaxation_spectrum"], "chi_avg")
            ω_arr =  read(f["relaxation_spectrum"], "omega_1")

            pX = plot(xscale=:log, yscale=:log)
            scatter!(pX, ω_arr[1:end], χ_avg[1:end],
                    markerstrokewidth = 0.0, label=false)
            
            # for s in 1:Ns 
            #     χs = read(f["relaxation_spectrum"], "chi_s_$(s)")
            #     ω_arr =  read(f["relaxation_spectrum"], "omega_$(s)")
            #     scatter!(pX, ω_arr[1:2:end], χs[1:2:end], label=false, 
            #     markershape=:+, alpha=0.5)
            # end

            # plot!(pX, ω_arr, 0.1.*ω_arr.^(a), label="ω^a")
            # plot!(pX, ω_arr, 0.0000008.*ω_arr.^(-b), label="ω^-b")
            # plot!(pX, ω_arr, 0.000035.*ω_arr.^(-a), label="ω^-a")
    
            χ_SB, ω_arr_SB = compute_relaxation_spectrum(t_arr[2:end], F_avg_stretched_exp_SB[2:end])

            χ_a, ω_arr_a = compute_relaxation_spectrum(t_arr[2:end], powerlaw_a.(t_arr[2:end],fit_params_a))

            χ_b, ω_arr_b = compute_relaxation_spectrum(t_arr[2:end], powerlaw_b.(t_arr[2:end],fit_params_b))

            plot!(pX, ω_arr_SB, χ_SB, ls=:dash, label="Kolsrauch")
            plot!(pX, ω_arr_a, χ_a, ls=:dash, label="a")
            plot!(pX, ω_arr_b, χ_b, ls=:dash, label="b")

            plot!(pX, xlims=(10.0^-15, 5), ylims=(10.0^-3, 0.5))
            plot!(pX, title="δ = $(round(δ, digits=3))", legend=:topleft)

            display(pX)

            close(f)
        end
    end

end

# Older version: fit a and b directly from |F̄_s - f̄_c| in log-log scale (kept for reference)
# for sol_file in mct_sol_files

#     f = h5open(joinpath(dir, sol_file), "r")

#     Ns = parse(Int64, split(sol_file, '_')[3])
#     δ = parse(Float64, split(sol_file, '_')[5])
#     ϵ = parse(Float64, split(sol_file, '_')[7][1:end-5])

#     tmin_a = 3*10.0^2
#     tmax_a = 5*10.0^3

#     tmin_b = 5*10.0^6
#     tmax_b = 8*10.0^6

#     if Ns == 5 && ϵ == minimum(ϵ_arr) 

#         println("For δ = $(δ)")

#         t_arr = read(f, "t_array")
#         Fs_avg = read(f["F_tagged"], "F_s_avg")

#         F_crit_avg = 0.0

#         for s in 1:Ns 
#             F_crit_avg += read(f["F_tagged_crit"], "F_s_c_$(s)")
#         end

#         F_crit_avg /= Ns

#         p = plot(xscale=:log10, yscale=:log10)

#         scatter!(p, t_arr[2:10:end], abs.(Fs_avg .- F_crit_avg)[2:10:end], 
#                 markerstrokewidth = 0.0, label=false)
#         plot!(p, title="δ = $(round(δ, digits=3))")
#         vline!(p, [tmin_a, tmin_b, tmax_a, tmax_b], 
#         color="black", linestyle=:dash, label=false)

#         indices_to_fit_a = [i for i in eachindex(t_arr) if tmin_a < t_arr[i] < tmax_a]
#         indices_to_fit_b = [i for i in eachindex(t_arr) if tmin_b < t_arr[i] < tmax_b]

#         to_fit_a = abs.(Fs_avg .- F_crit_avg)[indices_to_fit_a]
#         t_to_fit_a = t_arr[indices_to_fit_a]

#         to_fit_b = abs.(Fs_avg .- F_crit_avg)[indices_to_fit_b]
#         t_to_fit_b = t_arr[indices_to_fit_b]

#         scatter!(p, t_to_fit_a, to_fit_a, label=false, markerstrokewidth = 0.0)
#         scatter!(p, t_to_fit_b, to_fit_b, label=false, markerstrokewidth = 0.0)

#         powerlaw(t, p) = log10(abs(p[1])) .- p[2].*log10.(t)

#         p0_a = [rand(), rand()]
#         p0_b = [rand(), rand()]

#         fit_a = curve_fit(powerlaw, t_to_fit_a, log10.(to_fit_a), p0_a)
#         fit_params_a = [fit_a.param[1], fit_a.param[2]]

#         fit_b = curve_fit(powerlaw, t_to_fit_b, log10.(to_fit_b), p0_b)
#         fit_params_b = [fit_b.param[1], fit_b.param[2]]

#         a = fit_params_a[2]
#         b = fit_params_b[2]

#         println("fitted (a,b)=($(a), $(b))")

#         plot!(p, t_arr, abs(fit_params_a[1]).*t_arr.^(-fit_params_a[2]))

#         plot!(p, t_arr, abs(fit_params_b[1]).*t_arr.^(-fit_params_b[2]))

#         plot!(p, xlims=(10.0^-3, 10.0^15), ylims=(0.5*10.0^-3, 10.0^-1))
        
#         val_a = gamma(1-a)^2/gamma(1-2*a)
#         val_b = gamma(1-b)^2/gamma(1-2*b)
#         rel_arr = abs(val_a-val_b)/val_b

#         println("Relative Error Gamma relation fitted = $(rel_arr)") 

        # to_solve_b(bb) = val_a - gamma(1+bb)^2 / gamma(1+2*bb)
        # to_solve_a(aa) = gamma(1-aa)^2 / gamma(1-2*aa) - val_b
        # b_rel = find_zero(to_solve_b, (-1.0, 0.0))
        # a_rel = find_zero(to_solve_a, (0.0, 1.0))

        # println("b = $(b_rel) [obtained from Gamma relation with fitted a = $(a)]")
        # println("a = $(a_rel) [obtained from Gamma relation with fitted b = $(b)")

        # linear(t, p) = log10(abs(p[1])) .+ b_rel.*log10.(t)

        # p0 = [rand()]
        # fit_lin = curve_fit(linear, t_to_fit_b, log10.(to_fit_b), p0)
        # fit_params_lin = [fit_b.param[1]]

        # plot!(p, t_arr, 5*abs(fit_params_lin[1]).* t_arr.^(b_rel), ls=:dashdot)

#         display(p)

#         χ_avg = read(f["relaxation_spectrum"], "chi_avg")
#         ω_arr =  read(f["relaxation_spectrum"], "omega_1")

#         ω_min_a = 10.0^-4
#         ω_max_a = 10.0^-3

#         ω_min_b = 10.0^-9
#         ω_max_b = 10.0^-8

#         χ_indices_to_fit_a = [i for i in eachindex(ω_arr) if ω_min_a < ω_arr[i] < ω_max_a]
#         χ_indices_to_fit_b = [i for i in eachindex(ω_arr) if ω_min_b < ω_arr[i] < ω_max_b]

#         @show χ_indices_to_fit_b

#         χ_to_fit_a = χ_avg[χ_indices_to_fit_a]
#         ω_to_fit_a = ω_arr[χ_indices_to_fit_a]

#         χ_to_fit_b = χ_avg[χ_indices_to_fit_b]
#         ω_to_fit_b = ω_arr[χ_indices_to_fit_b]

#         pX = plot(xscale=:log, yscale=:log)
#         scatter!(pX, ω_arr[1:2:end], χ_avg[1:2:end], markerstrokewidth = 0.0)

#         scatter!(pX, ω_to_fit_a, χ_to_fit_a, markerstrokewidth = 0.0)

#         scatter!(pX, ω_to_fit_b, χ_to_fit_b, markerstrokewidth = 0.0)


#         plot!(pX, ω_arr, 0.1*ω_arr.^a, ls=:dash, label="ω^a", alpha=0.5)
#         plot!(pX, ω_arr, 0.00001.*ω_arr.^b, ls=:dash, label="ω^b", alpha=0.5)
        
#         β_SB = read(f["KWW_EXP"], "beta_SB")
#         τ_α = find_relaxation_time(t_arr,Fs_avg; threshold=F_crit_avg/exp(1))

#         F_avg_stretched_exp_SB = F_crit_avg .* exp.(.-(t_arr./τ_α).^β_SB)

#         χ_SB, ω_arr_SB = compute_relaxation_spectrum(t_arr[2:end], F_avg_stretched_exp_SB[2:end])

        
#         plot!(pX, ω_arr_SB, χ_SB)
        
#         plot!(pX, xlims=(10.0^-15, 10.0^5), ylims=(10.0^-3, 0.5))
#         plot!(pX, title="δ = $(round(δ, digits=3))")


#         display(pX)
        
#     end
#     close(f)
# end