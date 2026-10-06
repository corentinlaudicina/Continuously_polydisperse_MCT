#=
Solve the multicomponent MCT equations for a polydisperse hard-sphere fluid (discretised into
`Ns` species) at a given volume fraction, and save the tagged-particle correlation functions
and their relaxation spectra to an HDF5 file.

Usage:
    julia calculate_MCT_solution.jl Ns Δ ϕ key

    Ns:  number of species
    Δ:   width parameter of the size distribution (size ratio for key = "A3")
    ϕ:   volume fraction
    key: size distribution, "A3" or "gaussian"

Output: `mctsol_Ns_<Ns>_Delta_<Δ>_phi_<ϕ>.hdf5` in `Data_sol_<key>_epsilon/`.
=#

import Pkg; Pkg.activate(".")

using ModeCouplingTheory, HDF5, Dierckx

include("PercusYevick.jl")
include("QuantizeDistribution.jl")

"""
    find_d(N, Δ, μ, key)

Return the `N` species diameters for the size distribution `key` ("uniform", "gaussian" or "A3").
`μ` is the mean diameter and `Δ` the width parameter of the distribution
(for "A3", `Δ` acts as the size ratio and `μ` is unused).
"""
function find_d(N, Δ, μ, key)
    if (key == "uniform")
        d = find_uniform_quantization(μ, Δ, N)
        return d
    elseif (key == "gaussian")
        d = find_gaussian_quantization(μ,Δ,N)
        return d
    elseif (key == "A3")
        ## BEWARE FOR THIS Δ ACTS AS SIZE RATIO
        d = sample_A_sigma3(Δ, N)
        return d
    else
        error("DISTRIBUTION UNDEFINED")
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
    Nk, kmax: number of wave vectors and largest wave vector
    k_array: wave-vector grid, `k_i = (i - 1/2) kmax/Nk`
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
    Nk::T0
    kmax::T2
    k_array::T3
    Sk::T4
    kBT::T2
    m::T1
    k_peak_index::T0
    tmax ::T2
end

"""
    MCTParams(Δ, Ns, Nk, kmax, ϕ, t_max, key)

Build the [`MCTParams`](@ref) of an equimolar `Ns`-species mixture with size distribution `key`
and width parameter `Δ` at volume fraction `ϕ`, with unit mean diameter, `kBT = 1` and unit masses.
"""
function MCTParams(Δ, Ns, Nk, kmax, ϕ, t_max, key)
    μ = 1.0
    d = find_d(Ns, Δ, μ, key)
    x = [1/Ns for _ in 1:Ns]
    ρ_all = 6ϕ/(π*sum(x .* d .^3))
    ρ = x*ρ_all
    dk = kmax/Nk
    k_array = dk*(collect(1:Nk) .- 0.5)
    @show k_array
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

    return MCTParams(ϕ, d, Ns, x, ρ_all, ρ, Nk, kmax, k_array, Sk, kBT, m, k_peak_id, tmax)
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
    my_integral(integrand1, integrand2, grid)

Riemann sum of `integrand1(u) * integrand2[i]` over the uniform `grid`, where `integrand1` is a
function and `integrand2` holds precomputed values on the grid.
"""
function my_integral(integrand1, integrand2, grid)
    du = grid[2] - grid[1]
    I = 0.0
    for (i,ui) in enumerate(grid)
        I += integrand1(ui)*integrand2[i]
    end
    return I*du
end

"""
    compute_relaxation_spectrum_trap(t, Fₜ)

Compute the relaxation spectrum `χ(ω) = ∫ G(u) ωe^u/(1 + ω²e^{2u}) du` of the correlation
function `Fₜ(t)`, where `u = log t` and `G(u) = -dF/du` (obtained from a spline).
The integral is evaluated with a Riemann sum on 10⁴ uniform points in `u`.

Returns `(χ, ω)`, evaluated on 300 log-spaced frequencies from 10⁻³⁰ to 10⁴.
"""
function compute_relaxation_spectrum_trap(t, Fₜ)
    u = log.(t)
    Fᵤ_spline = Spline1D(u, Fₜ)
    G(u) = -derivative(Fᵤ_spline, u)
    u_grid = range(minimum(u), maximum(u), length=10000)
    Gu = G.(u_grid)

    @fastmath integrand(ω, u) = (ω *exp(u)) / (1+ω^2*exp(u)^2)

    χ(ω) = my_integral(u -> integrand(ω, u), Gu, u_grid)
    ω_arr = 10 .^ range(-30, 4, length=300)
    χdata = χ.(ω_arr)
    return χdata, ω_arr
end


"""
    save_run(Ns, Nk, kmax, Δ, key, ϕ, dir_to_save)

Run the collective and tagged MCT calculations for one state point and write the results
to `dir_to_save/mctsol_Ns_<Ns>_Delta_<Δ>_phi_<ϕ>.hdf5`, with groups

    run_params:          Nk, kmax, Ns, delta, phi, D_arr, distrib_type, k_peak
    F_tagged:            t_array and F_s_<s> (tagged correlation at the structure-factor peak)
    relaxation_spectrum: omega_arr and chi_s_<s>
"""
function save_run(Ns,Nk, kmax, Δ, key, ϕ, dir_to_save)
    t_max = 10.0^30
    params = MCTParams(Δ, Ns, Nk, kmax, ϕ, t_max, key)
    kindex = params.k_peak_index # max of summed collective Sk

    sol = solve_collective_MCT(params)

    ff = h5open(joinpath(dir_to_save,"mctsol_Ns_$(Ns)_Delta_$(Δ)_phi_$(ϕ).hdf5"), "w")

    create_group(ff, "F_tagged")
    create_group(ff, "relaxation_spectrum")
    create_group(ff, "run_params")

    write(ff["run_params"], "Nk", params.Nk)
    write(ff["run_params"], "kmax", params.kmax)
    write(ff["run_params"], "Ns", Ns)
    write(ff["run_params"], "delta", Δ)
    write(ff["run_params"], "phi", ϕ)
    write(ff["run_params"], "D_arr", params.d)
    write(ff["run_params"], "distrib_type", key)
    write(ff["run_params"], "k_peak", params.k_array[kindex])
    write(ff["F_tagged"], "t_array", sol.t)

    for s in 1:Ns
        tagged_sol_temp = solve_tagged_MCT(s, sol, params)
        Fs = getindex.(tagged_sol_temp.F, kindex)
        χ_arr, ω_arr = compute_relaxation_spectrum_trap(sol.t[2:end], Fs[2:end])

        write(ff["F_tagged"], "F_s_$(s)", Fs)
        write(ff["relaxation_spectrum"], "chi_s_$(s)", χ_arr)

        if s==1
            write(ff["relaxation_spectrum"], "omega_arr", ω_arr)
        end

    end
    close(ff)

end

# Command-line arguments (see header)
Ns = parse(Int64, ARGS[1])
Δ = parse(Float64, ARGS[2])
ϕ = parse(Float64, ARGS[3])
key = ARGS[4]

Nk = 100
kmax = 40.0

if key == "A3"
    dir_to_save = "Data_sol_A3_epsilon"
    save_run(Ns, Nk, kmax, Δ, key, ϕ, dir_to_save)
elseif key == "gaussian"
    dir_to_save = "Data_sol_gaussian_epsilon"
    save_run(Ns, Nk, kmax, Δ, key, ϕ, dir_to_save)
end
