#=
Plot the species-averaged tagged correlation function F̄_s(t) (`p`) and susceptibility χ̄''(ω)
(`pX`) for every MCT solution in `Data_sol_uniform/`, one colour per polydispersity δ.
Despite the file name, no relaxation time is computed.
=#

using Plots, DelimitedFiles, HDF5, LaTeXStrings, Statistics

plot_font = "Computer Modern"
default(fontfamily=plot_font,
        linewidth=2, framestyle=:box, label=nothing, grid=false)

dir = "Data_sol_uniform/"
files = readdir(dir)
p = plot(xaxis=:log, ylims=(0,1), xlims=(10.0^-3, 10.0^11))
pX = plot(xaxis=:log, yaxis=:log, 
          xlims=(10.0^-11, 10.0^3), 
          ylims=(10.0^-3, 1.0))


colmap = cgrad(:matter, length(files)+1, categorical = true)
count = 2
for sol in files

    f = h5open(joinpath(dir, sol), "r")

    Ns = read(f["run_params"], "Ns")
    
    t_array = read(f["F_tagged"], "t_array")
    ω_array = read(f["relaxation_spectrum"], "omega_arr")

    Fs_avg = zeros(length(t_array))
    Xs_avg = zeros(length(ω_array))

    for s in 1:Ns 
        Fs_avg .+= read(f["F_tagged"], "F_s_$(s)")
        Xs_avg .+= read(f["relaxation_spectrum"], "chi_s_$(s)")

      #   plot!(p, t_array[2:end], read(f["F_tagged"], "F_s_$(s)")[2:end], 
      #         linestyle=:dash, color=colmap[count], label=false, 
      #         alpha=0.2)

      #   plot!(pX, ω_array, read(f["relaxation_spectrum"], "chi_s_$(s)"), 
      #         linestyle=:dash, color=colmap[count], label=false, 
      #         alpha=0.2)

    end
    Fs_avg ./= Ns
    Xs_avg ./= Ns

    Δ = read(f["run_params"], "delta")

    D_arr = collect(read(f["run_params"], "D_arr"))
    δ = std(D_arr, corrected=false)

    @show Δ, δ
    plot!(p, t_array[2:end], Fs_avg[2:end], 
          color=colmap[count], label=L"$\delta = %$(round(δ, digits=3))$")
    plot!(pX, ω_array, Xs_avg, color=colmap[count], 
          label=L"$\delta = %$(round(δ, digits=3))$")
    close(f)
    count += 1
end

plot!(p, legend=:topright)

display(p)
display(pX)

# Older analyses on the previous JLD2 layout (Data/mctsol_..._eps_<ϵ>.jld2), kept for reference:
# ϵ_array = readdlm("Data/epsilon_array.txt")
# Δ_array = readdlm("Data/Delta_array.txt")
# iΔ = 27
# Δ = Δ_array[iΔ]
# Ns = 5

# Fs_sigma
# iΔ_array = [6, 17, 27]
# iϵlist = [1, 25, 50, 75]

# Δ = Δ_array[27]
# p_arr = []
# ϵ = ϵ_array[1]
# Ns = 10
# for iΔ in iΔ_array
#     count = 0
#     Δ = Δ_array[iΔ]
#     p = plot(xaxis=:log, xlabel=L"$t$", ylabel=L"$F_{s}(k,t)$", 
#             # xtickfont=font(18), 
#             # ytickfont=font(18), 
#             guidefont=font(11), 
#             legendfont=font(10))
#     filename = "Data/mctsol_Ns_$(Ns)_Delta_$(Δ)_eps_$(ϵ).jld2"
#     t_array = jldopen(filename)["t_array"]
#     Fs_avg = zeros(length(t_array))
#     for s = 1:Ns
#         if count == 0
#             Fs = jldopen(filename)["F_tagged"]["F_s_$(s)"]
#             Fs_avg .+= Fs./Ns
#             plot!(p, t_array[2:end], Fs[2:end], label=L"F_s^{(\sigma)(k,t)}", 
#             ls=:dash, linewidth=1.0, color="blue", alpha=0.5)
#             count+=1
#         else
#             Fs = jldopen(filename)["F_tagged"]["F_s_$(s)"]
#             Fs_avg .+= Fs./Ns
#             plot!(p, t_array[2:end], Fs[2:end], label=false, 
#             ls=:dash, linewidth=1.0, color="blue", alpha=0.5)
#         end
#     end
#     plot!(p, t_array[2:end], Fs_avg[2:end], linewidth=2.0, color="blue",
#     label=L"$\overline{F}_s(k,t)$")
#     plot!(p, xlims=(10.0^-3, 10^13), minorticks=5,
#      title=L"δ= %$(round(Δ, digits=5))",titlefontsize=13)
#     hline!(p, [1/exp(1)], color="black", ls=:dash, lw=0.75, alpha=0.75)
#     hline!(p, [0/exp(1)], color="black", lw=1, alpha=0.75)
#     # plot!([0.05,1],[0.75,0.85],arrow=true,color=:black,linewidth=2,label="")
#     plot!(p,legend=:topright, foreground_color_legend = "black")
#     push!(p_arr, p)
#     savefig(p, "Plots/Fs_traj_Ns_$(Ns)_delta_$(Δ).svg")
#     display(p)
# end

# plot(p_arr[1],p_arr[2],p_arr[3], layout=(1,3), size = (1200, 400))

# for iΔ in eachindex(iΔ_array)
#     Δ = iΔ_array[iΔ]
#     ϵ = ϵ_array[1]
#     p = plot(legend=:topright,
#         xlims=(-5, 20),
#         xlabel=L"\log_{10}(t)",
#         ylabel=L"F_s^{(\alpha)}(t)")
#     filename = "Data/mctsol_Ns_$(Ns)_Delta_$(Δ)_eps_$(ϵ).jld2"
#     for s = 1:Ns
#         t_array = jldopen(filename)["t_array"]
#         Fs = jldopen(filename)["F_tagged"]["F_s_$(s)"]
#         plot!(p, log10.(t_array), Fs, label=L"\alpha=%$s")
#     end
#     display(p)
# end

# Collective Fs
# Ns = 5
# Δ = Δ_array[17]
# p2 = plot(
#     legend=:bottomright,
#     xlims=(-5, 15),
#     xlabel=L"\log_{10}(t)",
#     ylabel=L"F_s(t)")
# for iϵ in 1:25:length(ϵ_array)
#     ϵ = ϵ_array[iϵ]

#     filename = "Data/mctsol_Ns_$(Ns)_Delta_$(Δ)_eps_$(ϵ).jld2"
#     Fs = jldopen(filename)["F_tagged"]["F_s_$(1)"] / Ns
#     t_array = jldopen(filename)["t_array"]
#     Fs_crit_avg = jldopen(filename)["F_tagged_crit"]["F_s_c_$(1)"] / Ns

#     for s = 2:Ns
#         Fs .+= jldopen(filename)["F_tagged"]["F_s_$(s)"] / Ns
#         Fs_crit_avg += jldopen(filename)["F_tagged_crit"]["F_s_c_$(s)"] / Ns

#     end

#     Fs = Fs .- Fs_crit_avg

#     plot!(p2, log10.(t_array), log10.(abs.(Fs)), 
#     label=L"\epsilon = %$(round(ϵ, digits=5))")
#     hline!(p2, [log10(Fs_crit_avg)], ls=:dash)
#     plot!(title="Ns=$(Ns), δ=$(Δ)")
# end
# display(p2)
# savefig(p2, "Plots/critical_average_Fs_Ns_$(Ns)_delta_$(Δ).svg")

## Relaxation time per species
# τα_array = zeros(length(ϵ_array), Ns)
# p3 = plot(legend=:topright,
#             xlabel=L"\log_{10}(\epsilon)",
#             ylabel=L"\log_{10}(\tau_\alpha^{(\alpha)})"
#             )
# for iϵ in eachindex(ϵ_array)
#     ϵ = ϵ_array[iϵ]

#     filename = "Data/mctsol_Ns_$(Ns)_Delta_$(Δ)_eps_$(ϵ).jld2"
#     for s = 1:Ns
#         t_array = jldopen(filename)["t_array"]
#         Fs = jldopen(filename)["F_tagged"]["F_s_$(s)"]
#         τα = find_relaxation_time(t_array, Fs)
#         τα_array[iϵ, s] = τα
#     end
# end
# labels = [L"\alpha = %$α" for i=1:1, α = 1:Ns]
# scatter!(p3, log10.(ϵ_array), log10.(τα_array), labels=labels)

# display(p3)

# Relaxation time of mean Fs

# τα_array = zeros(length(ϵ_array))
# p4 = plot(legend=:topright,
#             xlabel=L"\log_{10}(\epsilon)",
#             ylabel=L"\log_{10}(\tau_\alpha)"
#             )
# for iϵ in eachindex(ϵ_array)
#     ϵ = ϵ_array[iϵ]

#     filename = "Data/mctsol_Ns_$(Ns)_Delta_$(Δ)_eps_$(ϵ).jld2"
#     Fs = jldopen(filename)["F_tagged"]["F_s_$(1)"] / Ns
#     t_array = jldopen(filename)["t_array"]

#     for s = 2:Ns
#         Fs .+= jldopen(filename)["F_tagged"]["F_s_$(s)"] / Ns
#     end
#     τα = find_relaxation_time(t_array, Fs)
#     τα_array[iϵ] = τα
# end
# scatter!(p4, log10.(ϵ_array), log10.(τα_array), label=nothing)
# display(p4)

# Relaxation spectrum per particle
# for iϵ in iϵlist
#     ϵ = ϵ_array[iϵ]
#     p = plot(title=L"\epsilon = %$(round(ϵ, digits=5))",
#         legend=:bottomright,
#         xlims=(-5, 20),
#         xlabel=L"\log_{10}(\omega)",
#         ylabel=L"\log_{10}(\chi^{\alpha}(\omega))")
#     for s = 1:Ns
#         filename = "Data/mctsol_Ns_$(Ns)_Delta_$(Δ)_eps_$(ϵ_array[iϵ]).jld2"
#         χ = jldopen(filename)["relaxation_spectrum"]["chi_s_$(s)"]
#         ω = jldopen(filename)["relaxation_spectrum"]["omega_$(s)"]
#         plot!(log10.(ω), log10.(χ), label="s = $s")
#     end
#     plot!(ylims=(-3, 0), xlims=(-14, 4))
#     display(p)
# end

# Total relaxation spectrum
# p5 = plot(
# legend=:bottomright,
# xlims=(-5, 20),
# xlabel=L"\log_{10}(\omega)",
# ylabel=L"\log_{10}(\chi(\omega))")
# for iϵ in 1:10:length(ϵ_array)
#     ϵ = ϵ_array[iϵ]
#     filename = "Data/mctsol_Ns_$(Ns)_Delta_$(Δ)_eps_$(ϵ_array[iϵ]).jld2"

#     χ = jldopen(filename)["relaxation_spectrum"]["chi_s_$(1)"]/Ns
#     ω = jldopen(filename)["relaxation_spectrum"]["omega_$(1)"]

#     for s = 1:Ns
#         χ .+= jldopen(filename)["relaxation_spectrum"]["chi_s_$(s)"]/Ns
#     end
#     plot!(p5, log10.(ω), log10.(χ), label=L"\epsilon = %$(round(ϵ, digits=5))", color=iϵ,lw=2)
# end
# plot!(p5, ylims=(-3, 0), xlims=(-14, 4))
# display(p5)


# Total relaxation spectrum with species
# p5 = plot(
# legend=:bottomright,
# xlims=(-5, 20),
# xlabel=L"\log_{10}(\omega)",
# ylabel=L"\log_{10}(\chi(\omega))")
# for iϵ in 1:25:length(ϵ_array)
#     ϵ = ϵ_array[iϵ]
#     filename = "Data/mctsol_Ns_$(Ns)_Delta_$(Δ)_eps_$(ϵ_array[iϵ]).jld2"

#     χ = jldopen(filename)["relaxation_spectrum"]["chi_s_$(1)"]/Ns
#     ω = jldopen(filename)["relaxation_spectrum"]["omega_$(1)"]

#     for s = 1:Ns
#         χ .+= jldopen(filename)["relaxation_spectrum"]["chi_s_$(s)"]/Ns
#         plot!(p5, log10.(ω), log10.(jldopen(filename)["relaxation_spectrum"]["chi_s_$(s)"]), ls=:dash, color=iϵ, label=false,lw=0.5)
#     end
#     plot!(p5, log10.(ω), log10.(χ), label=L"\epsilon = %$(round(ϵ, digits=5))", color=iϵ,lw=2)
# end
# plot!(p5, ylims=(-3, 0), xlims=(-14, 4))
# display(p5)


# include("find_relaxation_time.jl")

# p6 = plot(
# legend=:topright,
# # xlims=(-5, 20),
# xlabel=L"\log_{10}(\omega)",
# ylabel=L"\log_{10}(\chi(\omega))")
# for iΔ in iΔ_array
# # for iϵ in 1:25:length(ϵ_array)
#     ϵ = minimum(ϵ_array)
#     Δ = Δ_array[iΔ]
#     Ns = 10
#     filename = "Data/mctsol_Ns_$(Ns)_Delta_$(Δ)_eps_$(ϵ).jld2"
#     # χ = jldopen(filename)["relaxation_spectrum"]["chi_s_$(1)"]/Ns
#     ω = jldopen(filename)["relaxation_spectrum"]["omega_$(1)"]
#     t_array = jldopen(filename)["t_array"]
#     χ = zeros(length(ω))
#     Fs = zeros(length(t_array))
    
#     for s = 1:Ns
#         #@show typeof(jldopen(filename)["relaxation_spectrum"]["chi_s_$(s)"]./Ns)
#         χ .+= jldopen(filename)["relaxation_spectrum"]["chi_s_$(s)"]./Ns
#         Fs .+= jldopen(filename)["F_tagged"]["F_s_$(s)"]./Ns
#         # β = jldopen(filename)["KWW_exponent"]["beta_s_$(s)"]
#         # Fs_crit = jldopen(filename)["F_tagged_crit"]["F_s_c_$(s)"]
#         # τ_α = find_relaxation_time(t_array, jldopen(filename)["F_tagged"]["F_s_$(s)"] ; threshold=Fs_crit/exp(1))
#         # println("β=$(β)")
#         # println("Fs_crit=$(Fs_crit)")
#         # stretched_exp_sol = Fs_crit .* exp.(-(t_array ./τ_α).^(β))
#         # χs = compute_susceptibility(t_array[2:end], stretched_exp_sol[2:end])
#         # plot!(p5, log10.(ω), log10.(jldopen(filename)["relaxation_spectrum"]["chi_s_$(s)"]), ls=:dash, color=iϵ, label=false,lw=0.5)
#         # plot!(p6, log10.(ω), log10.(χs), ls=:dash, color=iϵ, label=false,lw=0.5)
#     end
#     @show τ_α_avg = find_relaxation_time(t_array, Fs)
#     plot!(p6, log10.(ω), log10.(χ), label=L"\delta = %$(round(Δ, digits=5))",lw=2)
#     plot!(title=L"N_s=%$(Ns)")
# end
# plot!(p6, ylims=(-3, 0), xlims=(-15, 4))
# display(p6)

# using Plots
# using Dierckx, QuadGK, Plots

# function compute_relaxation_spectrum_temp2(Fₜ, t)
#     # t = soltagged.t[2:end]
#     # kindex = params.k_peak_index # max of summed collective Sk
#     # Fₜ = getindex.(soltagged.F[2:end], kindex)  
#     u = log.(t)
#     Fᵤ_spline = Spline1D(u, Fₜ)
#     G(u) = -derivative(Fᵤ_spline, u)

#     println("Spline Interpolation OK")
#     integrand(ω, u) = G(u) * (ω *exp(u)) / (1+ω^2*exp(u)^2)
#     χ(ω) = quadgk(u -> integrand(ω, u), log10(minimum(t)), log10(maximum(t)))[1]
#     ω_arr = 10 .^ range(-30, 4, length=300)
#     χdata = χ.(ω_arr)
#     println("Transform Computation OK")
#     # jldopen(filename, "a+") do f
#     #     f["relaxation_spectrum"]["chi_s_$(s)"] = χdata
#     #     f["relaxation_spectrum"]["omega_$(s)"] = ω_arr
#     # end
#     return ω_arr,χdata
# end


# t_array = 10 .^(range(-5,10,length=1000))
# β_array = range(0.5, 1, length=10)

# stretched_exp_arr = []
# p = plot(xscale=:log)
# pX = plot(xscale=:log, yscale=:log)
# for i in 1:length(β_array)
#     temp = exp.(-(t_array./100).^β_array[i])
#     ω_arr,χdata = compute_relaxation_spectrum_temp2(temp, t_array)
#     scatter!(p, t_array, temp)
#     scatter!(pX, ω_arr, χdata, label="β=$(β_array[i])")
#     push!(stretched_exp_arr, temp)
# end
# plot!(pX, ylims=(10^-4, 1), xlims=(10^-6, 10))
# plot!(pX, legend=:bottomright)
# display(p)
# display(pX)
