#=
Helper functions included by find_critical_exponents.jl and find_stretched_exp_params.jl:
an older version of the MCT solver that writes its results to JLD2 files, and the relaxation
spectrum χ''(ω).

Despite the file name, `find_relaxation_time` itself is provided by ModeCouplingTheory.jl.
=#

using JLD2
using ModeCouplingTheory
using Dierckx, QuadGK

include("../PercusYevick.jl")
include("../QuantizeDistribution.jl")

"""
    find_d(Ns, Δ, key)

Return the `Ns` species diameters for the size distribution `key` ("uniform", "gaussian" or "A3")
with unit mean diameter and width parameter `Δ`.
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
        d = sample_A_sigma3.(Δ, N)
        d ./ mean(d)
        return d

    else 
        error("distribution undefined")
    end
end

"""
    run_full_MCT_procedure(ϵ, Δ, Ns, filename)

Solve the collective and all tagged MCT equations at distance `ϵ` from the critical point and
write the tagged correlation functions and their relaxation spectra to the JLD2 file `filename`.
"""
function run_full_MCT_procedure(ϵ, Δ, Ns, filename)
    params = MCTParams(ϵ, Δ, Ns)
    println("params initialised !")
    sol = solve_collective_MCT(filename, params)
    for s = 1:Ns
        soltagged = solve_tagged_MCT(s, sol, filename, params)
        compute_relaxation_spectrum(soltagged, filename, params, s)
    end
end

"""
    MCTParams

Container for all the inputs of an MCT calculation: volume fraction `ϕ`, diameters `d`, number of
species `Ns`, number fractions `x`, total and partial number densities `ρ_all` and `ρ`, wave vectors
`k_array`, PY structure factors `Sk`, `kBT`, masses `m`, and the wave-vector index `k_peak_index` at
which the tagged correlation functions are stored.
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
end

"""
    find_critical_volume_fraction(Δ, Ns, datafolder)

Bracket the critical volume fraction for width parameter `Δ` and `Ns` species, using the files
`..._Ns_<Ns>_poly_<Δ>_phi_<ϕ>.jld2` in `datafolder` (older layout, non-ergodicity parameter under
the key `fc`). A state is considered glassy when the sum of the non-ergodicity parameter exceeds 10⁻⁴.

Returns `(ϕ_low, ϕ_hi)`.
"""
function find_critical_volume_fraction(Δ, Ns, datafolder)

    files = readdir(datafolder)
    println("$(length(files)) files found")
    files = files[contains.(files, "poly_$(Δ)_")]
    files = files[contains.(files, "Ns_$(Ns)_")]
    mydict = Dict{Float64, Float64}()
    for file in files
        jldopen(joinpath(datafolder, file)) do f
            fc = f["fc"]
            ϕ = parse(Float64, split(file, '_')[7][1:end-5])
            mydict[ϕ] = sum(sum(fc))
        end
    end
    sorted_ϕ_arr = sort(collect(keys(mydict)))
    for (i, ϕ) in enumerate(sorted_ϕ_arr)
        fc = mydict[ϕ]
        if fc > 0.0001
            return sorted_ϕ_arr[i-1], sorted_ϕ_arr[i]
        end
    end
end

"""
    MCTParams(ϵ, Δ, Ns, key, dir_crit_data)

Build the [`MCTParams`](@ref) of an equimolar `Ns`-species mixture at volume fraction
`ϕ = ϕ_c (1 - ϵ)`, where ϕ_c is the midpoint of the bracket found in `dir_crit_data`. Uses 100 wave
vectors in [0.2, 39.8], `kBT = 1`, unit masses, and stores the tagged functions at k index 16.
"""
function MCTParams(ϵ, Δ, Ns, key, dir_crit_data)
    d = find_d(Ns, Δ, key)
    ϕlo, ϕhi = find_critical_volume_fraction(Δ, Ns, dir_crit_data) 
    ϕ =  (ϕhi+ϕlo)*(1 - ϵ)/2.0
    x = [1/Ns for _ in 1:Ns]
    ρ_all = 6ϕ/(π*sum(x .* d .^3))
    ρ = x*ρ_all
    k_array = range(0.2, 39.8, length=100)
    Sk = [SMatrix{Ns,Ns}(find_structure_factor_PY(k, d, ρ)) for k in k_array]
    kBT = 1.0
    m = ones(Ns)
    return MCTParams(ϕ, d, Ns, x, ρ_all, ρ, k_array, Sk, kBT, m, 16)
end


"""
    solve_collective_MCT(filename, params)

Solve the multicomponent MCT equation for the collective correlation matrices F_ij(k, t) up to
t = 10³⁰, write the time grid to the JLD2 file `filename` (and create its `F_tagged` and
`relaxation_spectrum` groups), and return the solution.
"""
function solve_collective_MCT(filename, params)
    Sinv = inv.(params.Sk)
    Sk = params.Sk
    Ns = params.Ns
    J = [SMatrix{Ns,Ns}(I(Ns) .* k^2 .* params.kBT .* params.x ./ params.m) for k in params.k_array]
    γ = [J[ik]*Sinv[ik] for ik in eachindex(params.k_array)]
    α = 1.0
    β = 0.0
    δ = Sk*0
    tmax = 10.0^30
    kernel = MultiComponentModeCouplingKernel(params.ρ, params.kBT, params.m, params.k_array, Sk)
    equation = MemoryEquation(α,β,γ,δ, Sk, Sk*0.0, kernel)
    solver = TimeDoublingSolver(; N=8, Δt = 10.0^-5, t_max = tmax, tolerance=10.0^-8, verbose=true)
    sol = ModeCouplingTheory.solve(equation, solver)
    
    jldopen(filename, "a+") do f
        f["t_array"] = sol.t
        JLD2.Group(f, "F_tagged")
        JLD2.Group(f, "relaxation_spectrum")

    end

    return sol 
end

"""
    solve_tagged_MCT(s, sol, filename, params)

Solve the tagged-particle MCT equation for species `s` using the collective solution `sol`, write
F_s(k, t) at `params.k_peak_index` to `filename` (`F_tagged/F_s_<s>`), and return the solution.
"""
function solve_tagged_MCT(s, sol, filename, params)
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
    kindex = params.k_peak_index # max of summed collective Sk
    jldopen(filename, "a+") do f
        f["F_tagged"]["F_s_$(s)"] = getindex.(taggedsol.F, kindex)
    end
    
    return taggedsol
end

"""
    solve_MCT_steady_state(params)

Return the collective non-ergodicity parameter (long-time limit of F_ij(k, t)).
"""
function solve_MCT_steady_state(params)
    Sinv = inv.(params.Sk)
    Sk = params.Sk
    Ns = params.Ns
    J = [SMatrix{Ns,Ns}(I(Ns) .* k^2 .* params.kBT .* params.x ./ params.m) for k in params.k_array]
    γ = [J[ik]*Sinv[ik] for ik in eachindex(params.k_array)]

    kernel = MultiComponentModeCouplingKernel(params.ρ, params.kBT, params.m, params.k_array, params.Sk)
    sol = solve_steady_state(γ,params.Sk,kernel)
    return sol 
end

"""
    solve_tagged_MCT_steady_state(s, params, sol)

Return the tagged non-ergodicity parameter of species `s`, given the collective one `sol`
(from [`solve_MCT_steady_state`](@ref)).
"""
function solve_tagged_MCT_steady_state(s, params, sol)
    m = params.m
    k_array = params.k_array
    kBT = params.kBT
    Sk = params.Sk

    Nk = length(k_array)
    γ = [k^2 * kBT  / m[s] for k in k_array]

    Fs0 = ones(Nk)
    taggedkernel = TaggedMultiComponentModeCouplingKernel(s, params.ρ, params.kBT, params.m, params.k_array, params.Sk, sol)

    tagged_steady_state = solve_steady_state(γ,Fs0,taggedkernel)

    return tagged_steady_state
end

"""
    compute_relaxation_spectrum(t, Fₜ)

Compute the relaxation spectrum `χ(ω) = ∫ G(u) ωe^u/(1 + ω²e^{2u}) du` of the correlation
function `Fₜ(t)`, where `u = ln t` and `G(u) = -dF/du` (obtained from a spline), by adaptive quadrature.

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
    compute_relaxation_spectrum(soltagged, filename, params, s)

Same as [`compute_relaxation_spectrum(t, Fₜ)`](@ref) for the tagged solution `soltagged` of
species `s` at `params.k_peak_index`, writing `chi_s_<s>` and `omega_<s>` to the JLD2 file `filename`.
"""
function compute_relaxation_spectrum(soltagged, filename, params, s)
    t = soltagged.t[2:end]
    kindex = params.k_peak_index # max of summed collective Sk
    Fₜ = getindex.(soltagged.F[2:end], kindex)
    u = log.(t)
    Fᵤ_spline = Spline1D(u, Fₜ)
    G(u) = -derivative(Fᵤ_spline, u)

    println("Spline Interpolation OK")
    integrand(ω, u) = G(u) * (ω *exp(u)) / (1+ω^2*exp(u)^2)
    χ(ω) = quadgk(u -> integrand(ω, u), minimum(u), maximum(u))[1]
    ω_arr = 10 .^ range(-30, 4, length=300)
    χdata = χ.(ω_arr)
    println("Transform Computation OK")
    jldopen(filename, "a+") do f
        f["relaxation_spectrum"]["chi_s_$(s)"] = χdata
        f["relaxation_spectrum"]["omega_$(s)"] = ω_arr
    end
end
