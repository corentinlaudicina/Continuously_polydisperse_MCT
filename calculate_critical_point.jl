#=
Locate the MCT glass-transition (critical) volume fraction ϕ_c of a polydisperse hard-sphere
mixture, discretised into `Ns` species, from the non-ergodicity parameter.

Usage:
    julia calculate_critical_point.jl Ns Δ μ key

    Ns:  number of species
    Δ:   width parameter of the size distribution (size ratio for key = "A3")
    μ:   mean diameter
    key: size distribution, "uniform", "gaussian" or "A3"

Every evaluation of the non-ergodicity parameter is written to
`Data_Phase_Diagram_Fine_Grid/fc_Ns_<Ns>_poly_<Δ>_phi_<ϕ>_<key>.jld2`.
=#

import Pkg; Pkg.activate(".")

using ModeCouplingTheory, Roots, JLD2

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
    find_non_ergodicity_parameter(ϕ, Ns, Δ, μ, key, dir)

Solve the steady-state MCT equations at volume fraction `ϕ` for the collective non-ergodicity
parameter matrices f_c(k) and the tagged-particle ones f_c^(s)(k) of every species, on a grid of
300 wave vectors up to k = 60.

The results (run parameters, structure factor, collective and tagged f_c) are saved to a JLD2 file
in `dir`. Returns the collective non-ergodicity parameter (one `Ns × Ns` matrix per wave vector).
"""
function find_non_ergodicity_parameter(ϕ, Ns, Δ, μ, key, dir)
    x = [1/Ns for _ in 1:Ns]

    d = sort(find_d(Ns, Δ, μ, key))

    @show d
    @assert all(d .> 0)
    @show (ϕ, Ns, Δ)

    ρ_all = 6ϕ/(π*sum(x .* d .^3))
    ρ = x*ρ_all

    kmax=60.0; Nk = 300; dk = kmax/Nk; k_array = dk*(collect(1:Nk) .- 0.5)

    Sk = [SMatrix{Ns,Ns}(find_structure_factor_PY(k, d, ρ)) for k in k_array]

    kBT = 1.0
    m = ones(Ns)
    Sinv = inv.(Sk)

    J = [SMatrix{Ns,Ns}(I(Ns) .* k^2 .* kBT .* x ./ m) for k in k_array]
    γ = [J[ik]*Sinv[ik] for ik in eachindex(k_array)]

    kernel = MultiComponentModeCouplingKernel(ρ, kBT, m, k_array, Sk)
    fc = solve_steady_state(γ, Sk, kernel, verbose=false, tolerance=10^-10) ;

    fc_to_save = fc.F[1]

    δ = std(d ; corrected=false)

    filename = dir*"/fc_Ns_$(Ns)_poly_$(Δ)_phi_$(ϕ)_"*key*".jld2"

    f = jldopen(filename, "w")

    JLD2.Group(f, "run_params")
    JLD2.Group(f, "non_erg_param")

    f["run_params"]["Ns"] = Ns
    f["run_params"]["D_arr"] = d
    f["run_params"]["delta_var"] = δ
    f["run_params"]["Delta"] = Δ
    f["run_params"]["distribution_type"] = key
    f["non_erg_param"]["fc_collective"] = fc_to_save
    f["structure_factor"] = Sk

    for s in 1:Ns
        Fs0 = ones(Nk)
        γs = [k^2 * kBT  / m[s] for k in k_array]
        taggedkernel = TaggedMultiComponentModeCouplingKernel(s, ρ, kBT, m, k_array, Sk, fc)
        fc_s = solve_steady_state(γs,Fs0,taggedkernel).F[1]

        f["non_erg_param"]["f_c_$(s)"] = fc_s
    end
    close(f)

    return fc_to_save
end

"""
    objective(ϕ, Ns, Δ, μ, key, dir)

Root-finding objective `Σ_k Σ_ij f_c,ij(k)² - 10⁻⁴`: it changes sign at the volume fraction where
the non-ergodicity parameter jumps from zero (fluid) to a finite value (glass).
"""
function objective(ϕ, Ns, Δ, μ, key, dir)
    F = find_non_ergodicity_parameter(ϕ, Ns, Δ, μ, key, dir)
    return sum( sum(getindex.(F, i, j).^2 for i in 1:Ns, j in 1:Ns) ) - 10^-4
end

"""
    find_critical_point(Ns, Δ, μ, key, dir)

Find by bisection on ϕ ∈ (0.5, 0.55) the critical volume fraction ϕ_c of the mixture, with
tolerance 10⁻⁶. Intermediate non-ergodicity parameters are saved in `dir`.
"""
function find_critical_point(Ns, Δ, μ, key, dir)

    ϕ_c = find_zero(ϕ -> objective(ϕ, Ns, Δ, μ, key, dir), (0.5, 0.55), xatol=10^-6)

    @show Δ, ϕ_c
    return ϕ_c
end

# Command-line arguments (see header)
Ns = parse(Int64, ARGS[1])
Δ =  parse(Float64, ARGS[2])
μ = parse(Float64, ARGS[3])
key = ARGS[4]

dir = "Data_Phase_Diagram_Fine_Grid"
find_critical_point(Ns, Δ, μ, key, dir)
