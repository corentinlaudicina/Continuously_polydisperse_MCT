#=
Percus-Yevick (PY) closure for multicomponent hard-sphere mixtures.

Provides the direct correlation function c_ij(K) and the partial structure
factor matrix S_ij(K), which are the static inputs of the multicomponent
mode-coupling theory (MCT) calculations in this folder.
=#

import Pkg; Pkg.activate(".")

using LinearAlgebra, QuadGK
using StaticArrays


"""
    find_direct_correlation_function_PY(K, diameters, ρ)

Return the exact solution of the Percus-Yevick approximation to the direct correlation function
c_ij(K) of a multicomponent hard-sphere mixture, as a `p × p` matrix (`p` = number of species).

The solution is obtained through Baxter's factorisation: the Fourier transform `Q̃` of the
Baxter function `q_ij(r)` is computed by numerical quadrature, and `C = I - Q̃'Q̃` (rescaled
by `sqrt(ρ_i ρ_j)`).

ARGS:
    K: wave number (Float64)
    diameters: Vector with diameters of the different species
    ρ: Vector with the number densities of the different species

the vectors `diameters` and `ρ` must have the same length.

ref: Baxter, R.J. Ornstein–Zernike Relation and Percus–Yevick Approximation for Fluid Mixtures, J. Chem. Phys. 52, 4559 (1970)
"""
function find_direct_correlation_function_PY(K, diameters, ρ)
    p = length(diameters)
    @assert p == length(ρ)
    # d[i,k]: contact distance between species i and k
    # s[i,k]: half the diameter difference
    d = (diameters .+ diameters')/2
    s = (diameters .- diameters')/2

    # ξ[ν] = π/6 Σ_j ρ_j d_jj^ν (moments of the size distribution; ξ[3] is the packing fraction)
    ξ = [π/6*sum(ρ[j] * d[j,j]^ν for j = 1:p) for ν in 1:3]
    # coefficients of the quadratic Baxter function q_ik(r), for r ∈ [s_ik, d_ik]
    a = [(1 - ξ[3])^(-2)*(1 - ξ[3] + 3*ξ[2]*d[i,i]) for i=1:p]
    b = [-3/2*d[i,i]^2*(1-ξ[3])^(-2)*ξ[2] for i=1:p]
    q(r, i,k ) = 1/2*a[i]*(r^2-d[i,k]^2)+b[i]*(r-d[i,k])
    Q̃ = [I[i,k] - 2π*sqrt(ρ[i]*ρ[k])*quadgk(r -> q(r,i,k)*cis(K*r), s[i,k], d[i,k])[1] for i=1:p, k=1:p]

    C = I - real.(Q̃'*Q̃)
    C ./= sqrt.(ρ .* ρ')
    return C
end

"""
    find_structure_factor_PY(K, diameters, ρ)

Return the `p × p` matrix of partial structure factors S_ij(K) of a multicomponent
hard-sphere mixture in the Percus-Yevick approximation.

It is computed from the Ornstein-Zernike relation, `S⁻¹ = diag(1/x) - ρ_tot c(K)`, with
`x = ρ/sum(ρ)` the number fractions and `c` given by [`find_direct_correlation_function_PY`](@ref).

ARGS:
    K: wave number (Float64)
    diameters: Vector with diameters of the different species
    ρ: Vector with the number densities of the different species
"""
function find_structure_factor_PY(K, diameters, ρ)
    x = ρ/sum(ρ)
    c = find_direct_correlation_function_PY(K, diameters, ρ)
    p = length(ρ)
    δ = Matrix{Float64}(I,p,p)
    S = inv(δ./x  - sum(ρ)*c)
    return S
end
