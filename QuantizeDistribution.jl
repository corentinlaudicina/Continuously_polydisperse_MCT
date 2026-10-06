#=
Discretisation of continuous particle-size distributions into `N` species.

Each function returns the `N` diameters obtained by evaluating the quantile function of
the distribution at the mid-points of `N` equal-probability bins, so that every species
carries the same number fraction 1/N.
=#

using Distributions

"""
    find_gaussian_quantization(μ, σ, N)

Return `N` diameters sampling a Gaussian distribution of mean `μ` and standard deviation `σ`.
"""
function find_gaussian_quantization(μ, σ, N)
    dist = Distributions.Gaussian(μ, σ)
    x = quantile.(dist, range(1/(2N), 1-1/(2N), length=N))
    return x
end

"""
    find_uniform_quantization(μ, Δ, N)

Return `N` diameters sampling a uniform distribution on `[μ-Δ, μ+Δ]`.
"""
function find_uniform_quantization(μ, Δ, N)
    dist = Distributions.Uniform(μ-Δ, μ+Δ)
    x = quantile.(dist, range(1/(2N), 1-1/(2N), length=N))
    return x
end

"""
    sample_A_sigma3(σ_ratio, N)

Return `N` diameters sampling the size distribution whose probability density scales as σ⁻³
(inverse-transform sampling at the mid-points of `N` equal-probability bins).

`σ_ratio` is the size ratio of the distribution (it plays the role of `Δ` in the other samplers).
The constant `A` is chosen such that the probability density is normalised to 1.
"""
function sample_A_sigma3(σ_ratio, N)

    A = 1/2 + 1/(σ_ratio-1)
    σmin = (1 + σ_ratio)/(2*σ_ratio)
    P = LinRange(1/(2*N), 1-1/(2*N), N)
    σ = sqrt.(1 ./(σmin^-2 .- 2P./A))
    return σ ./ mean(σ)
end
