#=
Relaxation spectra (susceptibilities) χ''(ω) of correlation functions F(t), included by
plot_excess_wings.jl.

    compute_relaxation_spectrum(t, F)       adaptive quadrature (QuadGK)
    compute_relaxation_spectrum_trap(t, F)  Riemann sum on 10⁴ points (faster)
    write_susceptibility_to_file(dir_sol)   recompute and overwrite χ''_s for every MCT solution file in `dir_sol`

Both spectra use χ''(ω) = ∫ G(u) ωe^u/(1 + ω²e^{2u}) du with u = ln t and G(u) = -dF/du, i.e.
-dF/d ln t is treated as the distribution of relaxation times.
=#

# using DelimitedFiles, LaTeXStrings
using HDF5, CairoMakie
using Printf, Dierckx, QuadGK
# using LsqFit, SpecialFunctions, Roots, Measurements

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

    # println("Spline Interpolation OK")
    integrand(ω, u) = G(u) * (ω *exp(u)) / (1+ω^2*exp(u)^2)
    # function my_integral(integrand, a, b)
    #     u = range(a, b, length=1000)
    #     du = u[2] - u[1]
    #     return sum(integrand.(u))*du
    # end
    # χ(ω) = my_integral(u -> integrand(ω, u), log10(minimum(t)), log10(10.0^Δ))
    χ(ω) = quadgk(u -> integrand(ω, u), minimum(u), maximum(u))[1]
    ω_arr = 10 .^ range(-30, 4, length=300)
    χdata = χ.(ω_arr)
    # println("Transform Computation OK")
    return χdata, ω_arr
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

Same transform as [`compute_relaxation_spectrum`](@ref), with the integral evaluated by a Riemann
sum on 10⁴ uniform points in `u = ln t` (G(u) is computed once and reused for every ω).

Returns `(χ, ω)`, evaluated on 300 log-spaced frequencies from 10⁻³⁰ to 10⁴.
"""
function compute_relaxation_spectrum_trap(t, Fₜ)
    u = log.(t)
    Fᵤ_spline = Spline1D(u, Fₜ)
    G(u) = -derivative(Fᵤ_spline, u)
    u_grid = range(minimum(u), maximum(u), length=10000)
    Gu = G.(u_grid)


    # println("Spline Interpolation OK")
    @fastmath integrand(ω, u) = (ω *exp(u)) / (1+ω^2*exp(u)^2)

    χ(ω) = my_integral(u -> integrand(ω, u), Gu, u_grid)
    ω_arr = 10 .^ range(-30, 4, length=300)
    χdata = χ.(ω_arr)
    # println("Transform Computation OK")
    return χdata, ω_arr
end

"""
    write_susceptibility_to_file(dir_sol)

For every MCT solution file in `dir_sol`, recompute the susceptibility of each species from the
stored `F_tagged/F_s_<s>` with [`compute_relaxation_spectrum_trap`](@ref), and overwrite
`relaxation_spectrum/chi_s_<s>` and `relaxation_spectrum/omega_arr` in place.
"""
function write_susceptibility_to_file(dir_sol)
    sol_list = readdir(dir_sol)
    @show sol_list
    for sol in sol_list
        f = h5open(joinpath(dir_sol, sol), "r+")
        
        Ns = read(f["run_params"], "Ns")
        t_array = read(f["F_tagged"], "t_array")[2:end]

        for s in 1:Ns 
            Fs = read(f["F_tagged"], "F_s_$(s)")[2:end]

            # println("QUADGK")
            # @time χdataGK, ω_arrGK = compute_relaxation_spectrum(t_array, Fs)
            println("TRAP")
            @time χdataTR, ω_arrTR = compute_relaxation_spectrum_trap(t_array, Fs)

            if "chi_s_$(s)" in keys(f["relaxation_spectrum"])
                delete_object(f["relaxation_spectrum"], "chi_s_$(s)")
            end
            
            write(f["relaxation_spectrum"], "chi_s_$(s)", χdataTR)

            if s == 1
                delete_object(f["relaxation_spectrum"], "omega_arr")
                write(f["relaxation_spectrum"], "omega_arr", ω_arrTR)
            end

        end
        close(f)
    end

end

# Example: recompute the spectra of one data folder, then plot the species averages
# dir_sol = "Data_sol_uniform_epsilon_paper"
# sol_list = readdir(dir_sol)
# write_susceptibility_to_file(dir_sol)

# fig = Figure(resolution=(600,600))
# Makie.theme(:fonts).:regular = "CMU Serif"
# ax = Axis(fig[1, 1], ylabel=L"$\overline{\chi}_s''(k,\, t)$", xlabel=L"$\omega$", xscale=log10, yscale=log10, limits=(10.0^-11, 10.0^2, 5*10.0^-3, 0.5))

# for sol in sol_list
#     f = h5open(joinpath(dir_sol, sol), "r+")
#     Ns = read(f["run_params"], "Ns")

#     ω_arr = read(f["relaxation_spectrum"], "omega_arr")
#     χ_avg = zeros(length(ω_arr))
#     for s in 1:Ns 
#         χ_avg .+= read(f["relaxation_spectrum"], "chi_s_$(s)")
#     end
#     χ_avg ./ Ns
#     scatter!(ax, ω_arr, χ_avg)
#     close(f)

# end

# display(fig)