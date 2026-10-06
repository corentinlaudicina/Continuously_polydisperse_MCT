#=
Find, by bisection on the volume fraction ϕ, the state point at which the average tagged-particle
relaxation time τ_α of a polydisperse mixture equals a target value, then compute and save the
full MCT solution at that state point.

Usage:
    julia calculate_const_relax_time.jl Δ

    Δ: width parameter of the size distribution (size ratio for key = "A3")

Ns, key, the target τ_α and the integration time are set at the bottom of the file.
Output: `Data_sol/mctsol_Ns_<Ns>_Delta_<Δ>.hdf5`. Progress (τ_α for every ϕ visited by the
bisection) is logged in `Errors/progress_Delta_<Δ>_Ns_<Ns>.txt` and re-used on restart.
=#

import Pkg; Pkg.activate(".")

using ModeCouplingTheory, Roots, HDF5
using Dierckx, QuadGK

include("PercusYevick.jl")
include("QuantizeDistribution.jl")

"""
    find_d(Ns, Δ, key)

Return the `Ns` species diameters for the size distribution `key` ("uniform", "gaussian" or "A3")
with unit mean diameter and width parameter `Δ` (for "A3", `Δ` acts as the size ratio).
"""
function find_d(Ns, Δ, key)

    if (key == "uniform")
        mean_D = 1.0
        d = find_uniform_quantization(mean_D, Δ, Ns)
        return d
    elseif (key == "gaussian")
        mean_D = 1.0
        d = find_gaussian_quantization(mean_D, Δ, Ns)
        return d

    elseif (key == "A3")
        ## BEWARE FOR THIS Δ ACTS AS SIZE RATIO
        d = sample_A_sigma3(Δ, Ns)
        return d

    else
        error("distribution undefined")
    end
end

"""
    MCTParams

Container for all the inputs of an MCT calculation.

Fields:
    ϕ: volume fraction
    d: species diameters
    Ns: number of species
    x: number fractions of the species
    ρ_all: total number density
    ρ: number densities of the species
    k_array: wave-vector grid
    Sk: Percus-Yevick partial structure factor matrix at each wave vector
    kBT: thermal energy
    m: species masses
    k_peak_index: index of the maximum of the summed structure factor Σ_ij S_ij(k)
    tmax: final time of the MCT integration
"""
mutable struct MCTParams{T0, T1, T2, T3, T4}
    ϕ::T2
    d::T1
    Ns::T0
    x::T1
    ρ_all::T2
    ρ::T1
    k_array::T3
    Sk::T4
    kBT::T2
    m::T1
    k_peak_index::T0
    tmax ::T2
end

"""
    MCTParams(Δ, Ns, ϕ, t_max, key)

Build the [`MCTParams`](@ref) of an equimolar `Ns`-species mixture with size distribution `key`
and width parameter `Δ` at volume fraction `ϕ`, on a grid of 100 wave vectors in [0.2, 39.8],
with `kBT = 1` and unit masses.
"""
function MCTParams(Δ, Ns, ϕ, t_max, key)
    d = find_d(Ns, Δ, key)
    x = [1/Ns for _ in 1:Ns]
    ρ_all = 6ϕ/(π*sum(x .* d .^3))
    ρ = x*ρ_all
    k_array = range(0.2, 39.8, length=100)
    Sk = [SMatrix{Ns,Ns}(find_structure_factor_PY(k, d, ρ)) for k in k_array]
    kBT = 1.0
    m = ones(Ns)
    tmax = t_max

    # summed structure factor Σ_ij S_ij(k), used to locate the main peak
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

    return MCTParams(ϕ, d, Ns, x, ρ_all, ρ, k_array, Sk, kBT, m, k_peak_id, tmax)
end

"""
    solve_collective_MCT(params)

Solve the multicomponent MCT equation for the collective (coherent) correlation matrices
F_ij(k, t) with a time-doubling solver up to `params.tmax`. Returns the solution object.
"""
function solve_collective_MCT(params)
    Sinv = inv.(params.Sk)
    Sk = params.Sk
    Ns = params.Ns
    J = [SMatrix{Ns,Ns}(I(Ns) .* k^2 .* params.kBT .* params.x ./ params.m) for k in params.k_array]
    γ = [J[ik]*Sinv[ik] for ik in eachindex(params.k_array)]
    α = 1.0
    β = 0.0
    δ = Sk*0
    tmax = params.tmax
    kernel = MultiComponentModeCouplingKernel(params.ρ, params.kBT, params.m, params.k_array, Sk)
    equation = MemoryEquation(α,β,γ,δ, Sk, Sk*0.0, kernel)
    println("Solving MCT at ϕ = $(params.ϕ)")
    solver = TimeDoublingSolver(; N=8, Δt = 10.0^-5, t_max = tmax, tolerance=10.0^-8, verbose=false)
    sol = @time ModeCouplingTheory.solve(equation, solver)
    return sol
end

"""
    solve_tagged_MCT(s, sol, params)

Solve the MCT equation for the tagged-particle (incoherent) correlation function F_s(k, t)
of species `s`, using the collective solution `sol` (from [`solve_collective_MCT`](@ref))
in the memory kernel and the same solver settings.
"""
function solve_tagged_MCT(s, sol, params)
    m = params.m
    ρ = params.ρ
    k_array = params.k_array
    kBT = params.kBT
    Sk = params.Sk

    Nk = length(k_array)
    α = 1.0
    β = 0.0
    γ = [k^2 * kBT  / m[s] for k in k_array]
    δ = zeros(Nk)

    Fs0 = ones(Nk)
    dFs0 = zeros(Nk)

    taggedkernel = TaggedMultiComponentModeCouplingKernel(s, ρ, kBT, m, k_array, Sk, sol)
    taggedequation = MemoryEquation(α,β,γ,δ, Fs0, dFs0, taggedkernel)
    solver = sol.solver
    taggedsol = ModeCouplingTheory.solve(taggedequation, solver)
    return taggedsol
end


"""
    τ_α_has_been_calculated(ϕ, filename)

Look up in the progress log `Errors/<filename>` whether τ_α has already been computed at
volume fraction `ϕ`. Returns the stored τ_α, or `0.0` if none was found (the log file is
created empty if it does not exist yet).
"""
function τ_α_has_been_calculated(ϕ, filename)

    filename_list = readdir("Errors")
    τ_α = 0.0

    if filename in filename_list
        open(joinpath("Errors",filename), "r") do f
            for line in eachline(f)
                if contains(line, "ϕ = $ϕ")
                    τ_α += parse(Float64, split(line, ' ')[3][1:end-1])
                    println("read τ_α = $(τ_α) from file, ϕ = $ϕ")
                else continue
                end
            end
            if τ_α == 0.0
                println("τ_α not calculated for ϕ=$(ϕ), returning 0.")
            end
        end
        return τ_α
    else
        f = open(joinpath("Errors", filename), "w")
        close(f)
        return τ_α
    end
end

"""
    objective(ϕ, Δ, Ns, τ_α_target, t_max, key)

Root-finding objective `τ_α(ϕ) - τ_α_target`.

τ_α is the time at which the species-averaged tagged correlation function at the structure-factor
peak drops below 10⁻⁶ (capped at 10⁵⁰). It is read from the progress log if already known at `ϕ`,
otherwise computed from a full MCT solution up to `t_max` and appended to the log.
"""
function objective(ϕ, Δ, Ns, τ_α_target, t_max, key)
    filename = "progress_Delta_$(Δ)_Ns_$(Ns).txt"

    τ_α = τ_α_has_been_calculated(ϕ, filename)

    if τ_α == 0.0
        println("Calculating τ_α for ϕ = $(ϕ)")
        @assert t_max > τ_α_target
        params = MCTParams(Δ, Ns, ϕ, t_max, key)
        kindex = params.k_peak_index # max of summed collective Sk
        sol = solve_collective_MCT(params)
        t_array = sol.t
        tagged_sol = solve_tagged_MCT(1, sol, params)
        avg_tagged_sol = getindex.(tagged_sol.F, kindex)

        for s in 2:Ns
            tagged_sol_temp = solve_tagged_MCT(s, sol, params)
            avg_tagged_sol .+= getindex.(tagged_sol_temp.F, kindex)
        end

        avg_tagged_sol ./= Ns
        τ_α = min(10^50.0,find_relaxation_time(t_array, avg_tagged_sol ; threshold = 10.0^-6))

        list_progress_names = readdir("Errors")

        does_progress_file_exist = false

        for name in list_progress_names
            if name == "progress_Delta_$(Δ)_Ns_$(Ns).txt"
                does_progress_file_exist = true
            end
        end

        if does_progress_file_exist == true
            filename = "Errors/progress_Delta_$(Δ)_Ns_$(Ns).txt"
            open(filename, "a+") do io
                println(io, "τ_α = $(τ_α), ϕ = $ϕ")
            end
        else
            filename = "Errors/progress_Delta_$(Δ)_Ns_$(Ns).txt"
            open(filename, "w") do io
                println(io, "τ_α = $(τ_α), ϕ = $ϕ")
            end
        end
    end

    return τ_α-τ_α_target

end

"""
    find_const_relax_time(Δ, Ns, τ_α_target, tmax, key)

Find by bisection on ϕ ∈ (0.45, 0.55) the volume fraction at which τ_α equals `τ_α_target`.
"""
function find_const_relax_time(Δ, Ns, τ_α_target, tmax, key)
    f = ϕ -> objective(ϕ, Δ, Ns, τ_α_target, tmax, key)
    ϕ_c = find_zero(f, (0.45, 0.55), Roots.Bisection(), atol=10.0^8, verbose=true)
    return ϕ_c
end

"""
    compute_relaxation_spectrum(t, Fₜ)

Compute the relaxation spectrum `χ(ω) = ∫ G(u) ωe^u/(1 + ω²e^{2u}) du` of the correlation
function `Fₜ(t)`, where `u = ln t` and `G(u) = -dF/du` (obtained from a spline), by adaptive quadrature
over the whole time window.

Returns `(χ, ω)`, evaluated on 300 log-spaced frequencies from 10⁻³⁰ to 10⁴.
"""
function compute_relaxation_spectrum(t, Fₜ)
    u = log.(t)
    Fᵤ_spline = Spline1D(u, Fₜ)
    G(u) = -derivative(Fᵤ_spline, u)

    println("Spline Interpolation OK")
    integrand(ω, u) = G(u) * (ω *exp(u)) / (1+ω^2*exp(u)^2)
    χ(ω) = quadgk(u -> integrand(ω, u), minimum(u), maximum(u))[1]
    ω_arr = 10 .^ range(-30, 4, length=300)
    χdata = χ.(ω_arr)
    println("Transform Computation OK")
    return χdata, ω_arr
end

"""
    save_const_relax_time(Ns, Δ, τ_α_target, tmax, key, dir_to_save)

Locate the volume fraction at which τ_α = `τ_α_target` (see [`find_const_relax_time`](@ref)),
solve the MCT equations there up to t = 10³⁰ and write the results to
`dir_to_save/mctsol_Ns_<Ns>_Delta_<Δ>.hdf5`, with groups

    run_params:          Ns, delta, phi, D_arr, distrib_type, k_peak
    F_tagged:            t_array and F_s_<s> (tagged correlation at the structure-factor peak)
    relaxation_spectrum: omega_arr and chi_s_<s>
"""
function save_const_relax_time(Ns, Δ, τ_α_target, tmax, key, dir_to_save)

    println("Finding state point for Δ=$(Δ)")
    target_ϕ = find_const_relax_time(Δ, Ns, τ_α_target, tmax, key)
    println("Found state point for Δ=$(Δ) ! target_ϕ = $(target_ϕ)")

    tmax_new = 10.0^30
    params = MCTParams(Δ, Ns, target_ϕ, tmax_new, key)
    kindex = params.k_peak_index # max of summed collective Sk
    sol = solve_collective_MCT(params)

    ff = h5open(joinpath(dir_to_save,"mctsol_Ns_$(Ns)_Delta_$(Δ).hdf5"), "w")

    create_group(ff, "F_tagged")
    create_group(ff, "relaxation_spectrum")
    create_group(ff, "run_params")

    write(ff["run_params"], "Ns", Ns)
    write(ff["run_params"], "delta", Δ)
    write(ff["run_params"], "phi", target_ϕ)
    write(ff["run_params"], "D_arr", params.d)
    write(ff["run_params"], "distrib_type", key)
    write(ff["run_params"], "k_peak", params.k_array[kindex])

    write(ff["F_tagged"], "t_array", sol.t)

    for s in 1:Ns
        tagged_sol_temp = solve_tagged_MCT(s, sol, params)
        Fs = getindex.(tagged_sol_temp.F, kindex)
        χ_arr, ω_arr = compute_relaxation_spectrum(sol.t[2:end], Fs[2:end])

        write(ff["F_tagged"], "F_s_$(s)", Fs)
        write(ff["relaxation_spectrum"], "chi_s_$(s)", χ_arr)

        if s==1
            write(ff["relaxation_spectrum"], "omega_arr", ω_arr)
        end

    end
    close(ff)

end

# Search settings: integration time for the bisection and target relaxation time
tmax = 10.0^11
τ_α_target = 10.0^10
Ns = 10
Δ = parse(Float64, ARGS[1])

key = "A3"
dir = "Data_sol"

save_const_relax_time(Ns, Δ, τ_α_target, tmax, key, dir)
