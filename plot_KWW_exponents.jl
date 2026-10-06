#=
Plot the KWW stretching exponents stored in the MCT solutions of
`Data_sol_uniform_fixed_tau_alpha` (group `KWW_exponents`):
    left:  species-averaged β̄_KWW vs polydispersity δ
    right: per-species β_KWW vs diameter D, one colour per δ
           (species with β = 0, i.e. not fitted, are skipped)
=#

using CairoMakie, HDF5, LaTeXStrings
using ModeCouplingTheory

include("../QuantizeDistribution.jl")

"""
    find_d(Ns, Δ)

Return the `Ns` diameters of a uniform distribution of unit mean and half-width `Δ`.
"""
function find_d(Ns, Δ)
    mean_D = 1.0
    d = find_uniform_quantization(mean_D, Δ, Ns)
    return d
end

fig = Figure(resolution=(800,500))

Makie.theme(:fonts).:regular = "CMU Serif"

ax1 = Axis(fig[1, 1], ylabel=L"$\overline{\beta}_{\text{KWW}}$", xlabel=L"$\delta$")
ax2 = Axis(fig[1,2], xtickalign = 1, yaxisposition=:left, 
           ylabel=L"$\beta_{\text{KWW}}$", xlabel=L"$D$")
# ax_bottom = Axis(fig[2,2], xtickalign = 1, yaxisposition=:right, 
#            ylabel=L"$\beta_{\text{KWW}}$", xlabel=L"$D$")

rowgap!(fig.layout, 0)
# colgap!(fig.layout, 0)

fig

dir_MCT_sol = "Data_sol_uniform_fixed_tau_alpha"

sol_list = readdir(dir_MCT_sol)

colmap_ax1 = cgrad(:matter, 2, categorical = true)
colmap_ax2 = cgrad(:matter, length(sol_list), categorical = true)

δ_top = 0.1
δ_bottom = 0.5

# one file per δ
count = 1
for sol in sol_list
    f = h5open(joinpath(dir_MCT_sol, sol), "r")
    Ns = read(f["run_params"], "Ns")
    D_arr = read(f["run_params"], "D_arr")
    δ = std(D_arr ; corrected=false)
    β_KWW_avg = read(f["KWW_exponents"], "beta_KWW_avg")

    scatter!(ax1, [δ], [β_KWW_avg], color=colmap_ax1[2], 
             markerstrokewidth=0.0)

    for s in 1:Ns 
        βs = read(f["KWW_exponents"], "beta_KWW_s_$(s)")
        if βs != 0.0
            if s == 1 
                scatter!(ax2, [D_arr[s]], [βs], color=colmap_ax2[count], 
                        markerstrokewidth=0.0, 
                        label=L"$\delta = %$(round(δ, digits=3))$")
            else
                scatter!(ax2, [D_arr[s]], [βs], color=colmap_ax2[count], 
                        markerstrokewidth=0.0)
            end
        end
    end

    close(f)
    count += 1
end

fig

# Older Plots.jl version (KWW_EXP group of the previous file layout), kept for reference:
# plot_font = "Computer Modern"
# default(fontfamily=plot_font,
#         linewidth=2, framestyle=:box, label=nothing, grid=true)

# dir_MCT_sol = "Data_sol_temp/"

# Δ_arr, ϵ_arr, Ns_arr = get_params(dir_MCT_sol)
# ϵ_min = minimum(ϵ_arr)

# files_mct_sol = String[]
# for file in readdir(dir_MCT_sol)
#     if occursin("mctsol", file)
#         push!(files_mct_sol, file)
#     end
# end

# colmap = cgrad(:matter, 4, categorical = true)
# col=2

# p_exp = plot(xlabel=L"D",
#          ylabel=L"\beta_{KWW}")
    


# for file in files_mct_sol
#     Ns = parse(Int64, split(file, '_')[3])
#     Δ = parse(Float64, split(file, '_')[5])
#     d_arr = find_d(Ns, Δ)

#     f = h5open(joinpath(dir, file), "r")

#     β_avg_asymptotic = read(f["KWW_EXP"], "beta_asymptotic")
#     β_avg_SB = read(f["KWW_EXP"], "beta_SB")

    # hline!(p_exp, [β_avg_asymptotic], linestyle=:dash, 
    #        alpha=0.5, color="black", label=false)
    # hline!(p_exp, [β_avg_SB], linestyle=:dashdot, 
    #        alpha=0.5, color="red", label=false)

#     scatter!(p_exp, [d_arr[1]], [read(f["KWW_EXP"], "beta_asymptotic_s_1")], 
#     color=colmap[col], label=L"\delta=%$(round(Δ, digits=3))")

#     println()
#     for s in 2:Ns 
#         scatter!(p_exp, [d_arr[s]], [read(f["KWW_EXP"], "beta_asymptotic_s_$(s)")], 
#                     color=colmap[col], label=false, markershape=:circle)
#     end

#     for s in 1:Ns 
#         scatter!(p_exp, [d_arr[s]], [read(f["KWW_EXP"], "beta_SB_s_$(s)")], 
#         color=colmap[col], label=false, markershape=:rect)
#     end 
#     col += 1
#     close(f)
# end

# plot!(p_exp, ylims=(0.4, 0.9))
# display(p_exp)


# colmap = cgrad(:matter, 11, categorical = true)
# for file in files_mct_sol
#     Ns = parse(Int64, split(file, '_')[3])
#     Δ = parse(Float64, split(file, '_')[5])
#     d_arr = find_d(Ns, Δ)

#     f = h5open(joinpath(dir, file), "r")

#     t_arr = read(f, "t_array")[2:end]
#     p_F = plot(xaxis=:log)
#     col = 2
#     for s in 1:Ns 
#         Fs = read(f["F_tagged"], "F_s_$(s)")[2:end]
#         scatter!(p_F, t_arr[1:15:end], Fs[1:15:end], 
#         color=colmap[col], label=false, 
#         markersize=3)
#         plot!(p_F, title=L"\delta=%$(round(Δ, digits=3))")

#         β_asymptotic = read(f["KWW_EXP"],"beta_asymptotic_s_$(s)")
#         β_SB = read(f["KWW_EXP"],"beta_SB_s_$(s)")

#         F_crit = read(f["F_tagged_crit"], "F_s_c_$(s)")
#         τα = find_relaxation_time(t_arr, Fs ; threshold=F_crit/exp(1))
#         Fs_asymptotic_fit = F_crit .* exp.(.-(t_arr ./ τα).^β_asymptotic)
#         Fs_SB_fit = F_crit .* exp.(.-(t_arr ./ τα).^β_SB)

#         plot!(p_F, t_arr, Fs_asymptotic_fit, color=colmap[col], linestyle=:dash)
#         plot!(p_F, t_arr, Fs_SB_fit, color=colmap[col], linestyle=:dashdotdot)

#         col += 1
#     end
#     plot!(xlims=(10.0^6, 10.0^13))
#     display(p_F)
# end
