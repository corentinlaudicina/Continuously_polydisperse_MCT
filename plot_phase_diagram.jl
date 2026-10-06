#=
Phase-diagram and static-quantity figures for polydisperse hard spheres (Ns = 10 species), from
the non-ergodicity parameters computed by calculate_critical_point.jl.

    plot_Phase_Diagram()            δ vs ϕ_c for the uniform, Gaussian and inverse-cubic
                                    distributions, with sketches of the three distributions
    plot_Phase_Diagram_fine_grid()  same, comparing Nk = 100 (circles) and Nk = 300 (triangles)
    plot_avg_S()                    S̄(k) and f̄^(s)(k) at ϕ_c for several δ
                                    → Plots/Structure_factor_PY_polydispersity.pdf, Plots/LM_PY_polydispersity.pdf
    plot_LM()                       log f_1^(s)(k) of the smallest species for ϕ > ϕ_c
                                    (inverse cubic), one figure per Δ
    plot_LM_log()                   log f_σ^(s)(k) vs (kD̄)² at ϕ_c, uniform vs inverse cubic,
                                    δ ≈ 0.07 and δ ≈ 0.29 → Plots/log_LM.pdf

"LM" is the Lamb-Mössbauer factor f^(s)(k), i.e. the tagged non-ergodicity parameter.
Only `plot_Phase_Diagram()` is called when the file is run.
=#

using HDF5, JLD2
using Base.Threads, DelimitedFiles, CairoMakie
using LaTeXStrings, Statistics, Distributions
# using Makie.GeometryBasics
# datafolders = ["Data_Phase_Diagram_Gaussian", "Data_Phase_Diagram_Uniform_old", "Data_Phase_Diagram_Inverse_Cubic_old"]

"""
    find_critical_volume_fraction(Δ, Ns, key, datafolder)

Bracket the critical volume fraction for width parameter `Δ`, `Ns` species and distribution `key`
using the `fc_Ns_<Ns>_poly_<Δ>_phi_<ϕ>_<key>.jld2` files in `datafolder`. A state is considered
glassy when the sum of the collective non-ergodicity parameter exceeds 10⁻⁴.

Returns `(δ, ϕ_low, ϕ_hi)`: the polydispersity, the largest volume fraction without a glass
solution and the smallest one with it.

Note: here δ is the standard deviation of the particle *volumes* πD³/6, not of the diameters as in
the other scripts.
"""
function find_critical_volume_fraction(Δ, Ns, key, datafolder)

    files = readdir(datafolder)
    # println("$(length(files)) files found")
    
    files = files[contains.(files, "Ns_$(Ns)_")]
    files = files[contains.(files, key)]
    files = files[contains.(files, "poly_$(Δ)_")]

    mydict = Dict{Float64, Float64}()

    δ = 0
    count = 0
    # println("$(length(files)) files have been read in "*datafolder)
    # @show datafolder
    for file in files
        jldopen(joinpath(datafolder, file)) do f
            fc = f["non_erg_param"]["fc_collective"]
            # @show split(file, '_')[7]
            ϕ = parse(Float64, split(file, '_')[7])
            mydict[ϕ] = sum(sum(fc))

            if count == 0
                D_arr = f["run_params"]["D_arr"] 
                V_arr = 4/3 * π * (D_arr ./ 2).^3
                δ += std(V_arr ; corrected=false) #std(f["run_params"]["D_arr"] ; corrected=false)
                count +=1
            end
        end
    end
    sorted_ϕ_arr = sort(collect(keys(mydict)))

    for (i, ϕ) in enumerate(sorted_ϕ_arr)
        fc = mydict[ϕ]
        if fc > 0.0001
            return δ, sorted_ϕ_arr[i-1], sorted_ϕ_arr[i]
        end
    end
end

"""
    plot_Phase_Diagram()

Plot δ vs ϕ_c for the uniform, Gaussian and inverse-cubic size distributions (low-resolution scan
in `Data_Phase_Diagram`), with sketches of the three distributions on the right.
"""
function plot_Phase_Diagram()
    key_list = ["uniform", "gaussian", "A3"]

    num_phase_pts = 15 #30 for low resolution
    Ns = 10

    num_phase_pts_low_res = 30 

    Δ_arr_uniform_low_res = LinRange(0.1, 0.9, num_phase_pts_low_res)
    Δ_arr_gaussian_low_res = LinRange(0.05, 0.6, num_phase_pts_low_res)
    σ_ratio_arr_low_res = LinRange(1.0, 10, num_phase_pts_low_res)

    Δ_arr_uniform = LinRange(0.1, 0.9, num_phase_pts)
    Δ_arr_gaussian = LinRange(0.05, 0.6, num_phase_pts)
    σ_ratio_arr = LinRange(1.0, 10, num_phase_pts)

    # datafolder = "Data_Phase_Diagram_Fine_Grid"
    datafolder_low_res = "Data_Phase_Diagram"

    colmap = cgrad(:matter, 4, categorical = true)

    fig = Figure(resolution=(900,700))

    fontsize_theme = Theme(fontsize=30)
    set_theme!(fontsize_theme)

    Makie.theme(:fonts).:regular = "CMU Serif"

    ax = Axis(fig[1:4,1], 
                xlabel=L"$\varphi_c$", 
                ylabel=L"$\delta$", 
                limits=(0.513, 0.541, nothing, nothing), 
                xminorticksvisible=true, yminorticksvisible=true, 
                xgridvisible=false, ygridvisible=false)

    x = 0.0:0.001:2.0
    x3 = 1:0.001:2.0
    ax_r1 = Axis(fig[2,2], 
                    xticklabelsvisible=false, yticklabelsvisible=false, 
                    xticksvisible=false, yticksvisible=false,
                    limits=(0.5, 1.6, 0, 2), 
                    xlabel=L"$D$", ylabel=L"$P_{\text{uni.}}(D)$", 
                    xgridcolor = :white,
                    ygridcolor = :white)
    ax_r2 = Axis(fig[3,2], 
                    xticklabelsvisible=false, yticklabelsvisible=false, 
                    xticksvisible=false, yticksvisible=false,
                    limits=(0.4, 1.6, 0, 2), 
                    xlabel=L"$D$", ylabel=L"$P_{\text{Gauss}}(D)$",
                    xgridcolor = :white,
                    ygridcolor = :white)
    ax_r3 = Axis(fig[4,2], 
                    xticklabelsvisible=false, yticklabelsvisible=false, 
                    xticksvisible=false, yticksvisible=false,
                    limits=(0.8, 2.2, 0, 1.3), 
                    xlabel=L"$D$", ylabel=L"$P_{\text{inv.}}(D)$",
                    xgridcolor = :white,
                    ygridcolor = :white)

    x_pos_L, y_pos_L = 0.515, 0.4
    x_pos_G, y_pos_G = 0.525, 0.15
    text!(ax, x_pos_L, y_pos_L, text="Liquid")
    text!(ax, x_pos_G, y_pos_G, text="Ideal Glass")

    # sketches of the three size distributions
    lines!(ax_r1, x, pdf.(Uniform(0.7,1.4), x), color=colmap[3])
    band!(ax_r1, x, pdf.(Uniform(0.7,1.4), x), pdf.(Uniform(0.7,1.4), x) .- pdf.(Uniform(0.7,1.4), x), color=(colmap[3], 0.33))

    lines!(ax_r2, x, pdf.(Normal(1.0, 0.25), x), color=colmap[2])
    band!(ax_r2, x, pdf.(Normal(1.0, 0.25), x), pdf.(Normal(1.0, 0.25), x) .- pdf.(Normal(1.0, 0.25), x), color=(colmap[2], 0.33))

    lines!(ax_r3, x3, 1.0./x3.^3, color=colmap[4])
    band!(ax_r3, x3, 1.0./x3.^3, 1.0./x3.^3 .- 1.0./x3.^3, color=(colmap[4], 0.33))

    elem_1 = [MarkerElement(color = colmap[2], marker =:circle, markersize = 15)]
    elem_2 = [MarkerElement(color = colmap[3], marker =:circle, markersize = 15)]
    elem_3 = [MarkerElement(color = colmap[4], marker =:circle, markersize = 15)]

    Legend(fig[1,2], [elem_2, elem_1, elem_3], ["Uniform", "Gaussian", "Inverse Cubic"], 
            framevisible = false)
            
    # poly!(ax, Point2f[(0.5229, 0.4), (0.55, 0.4), (0.56, 0.525), (0.523, 0.525)], color = ("orange", 0.25), strokewidth = 0.0)
    # poly!(ax, Point2f[(0.5229, 0.46), (0.55, 0.46), (0.56, 0.525), (0.523, 0.525)], color = ("orange", 0.3), strokewidth = 0.0)
        
    for key in key_list
        if key == "gaussian"
            # for Δ in Δ_arr_gaussian
            #     δ, ϕc_low, ϕc_hi =  find_critical_volume_fraction(Δ, Ns, key, datafolder)

            #     scatter!(ax, [(ϕc_low+ϕc_hi)/2], [δ], color=colmap[2], marker=:utriangle, 
            #     label = false)

            # end

            for Δ in Δ_arr_gaussian_low_res
                δ, ϕc_low, ϕc_hi =  find_critical_volume_fraction(Δ, Ns, key, datafolder_low_res)

                scatter!(ax, [(ϕc_low+ϕc_hi)/2], [δ], color=colmap[2])

            end

        elseif key == "uniform"
            # for Δ in Δ_arr_uniform
            #     δ, ϕc_low, ϕc_hi =  find_critical_volume_fraction(Δ, Ns, key, datafolder)
            #     scatter!(ax, [(ϕc_low+ϕc_hi)/2], [δ], color=colmap[3], marker=:utriangle, 
            #     label = false)

            # end
            for Δ in Δ_arr_uniform_low_res
                δ, ϕc_low, ϕc_hi =  find_critical_volume_fraction(Δ, Ns, key, datafolder_low_res)

                scatter!(ax, [(ϕc_low+ϕc_hi)/2], [δ], color=colmap[3])

            end

        elseif key == "A3"
            # for Δ in σ_ratio_arr
            #     δ, ϕc_low, ϕc_hi =  find_critical_volume_fraction(Δ, Ns, key, datafolder)
            #     scatter!(ax, [(ϕc_low+ϕc_hi)/2], [δ],marker=:utriangle, color=colmap[4])
            # end
            for Δ in σ_ratio_arr_low_res
                δ, ϕc_low, ϕc_hi =  find_critical_volume_fraction(Δ, Ns, key, datafolder_low_res)
                scatter!(ax, [(ϕc_low+ϕc_hi)/2], [δ], color=colmap[4])
            end

        end
    end

    display(fig)
    # save("Plots/phase_diagram_polydisperse.pdf", fig)
end

plot_Phase_Diagram()

"""
    plot_Phase_Diagram_fine_grid()

Same as [`plot_Phase_Diagram`](@ref), overlaying the Nk = 100 scan (`Data_Phase_Diagram`, circles)
and the Nk = 300 scan (`Data_Phase_Diagram_Fine_Grid`, triangles) to check the k-grid convergence.
"""
function plot_Phase_Diagram_fine_grid()
    key_list = ["uniform", "gaussian", "A3"]

    num_phase_pts = 15 #30 for low resolution
    Ns = 10

    num_phase_pts_low_res = 30 

    Δ_arr_uniform_low_res = LinRange(0.1, 0.9, num_phase_pts_low_res)
    Δ_arr_gaussian_low_res = LinRange(0.05, 0.6, num_phase_pts_low_res)
    σ_ratio_arr_low_res = LinRange(1.0, 10, num_phase_pts_low_res)

    Δ_arr_uniform = LinRange(0.1, 0.9, num_phase_pts)
    Δ_arr_gaussian = LinRange(0.05, 0.6, num_phase_pts)
    σ_ratio_arr = LinRange(1.0, 10, num_phase_pts)

    datafolder = "Data_Phase_Diagram_Fine_Grid"
    datafolder_low_res = "Data_Phase_Diagram"

    colmap = cgrad(:matter, 4, categorical = true)

    fig = Figure(resolution=(700,500))

    fontsize_theme = Theme(fontsize=30)
    set_theme!(fontsize_theme)
    Makie.theme(:fonts).:regular = "CMU Serif"

    ax = Axis(fig[1,1], 
                xlabel=L"$\varphi_c$", 
                ylabel=L"$\delta$", 
                limits=(0.513, 0.541, 0.05, 0.49), 
                xgridvisible=false, ygridvisible=false)

    for key in key_list
        if key == "gaussian"
            for Δ in Δ_arr_gaussian
                δ, ϕc_low, ϕc_hi =  find_critical_volume_fraction(Δ, Ns, key, datafolder)

                scatter!(ax, [(ϕc_low+ϕc_hi)/2], [δ], color=colmap[2], marker=:utriangle, 
                label = false)

            end

            for Δ in Δ_arr_gaussian_low_res
                δ, ϕc_low, ϕc_hi =  find_critical_volume_fraction(Δ, Ns, key, datafolder_low_res)

                scatter!(ax, [(ϕc_low+ϕc_hi)/2], [δ], color=colmap[2])

            end

        elseif key == "uniform"
            for Δ in Δ_arr_uniform
                δ, ϕc_low, ϕc_hi =  find_critical_volume_fraction(Δ, Ns, key, datafolder)
                scatter!(ax, [(ϕc_low+ϕc_hi)/2], [δ], color=colmap[3], marker=:utriangle, 
                label = false)

            end
            for Δ in Δ_arr_uniform_low_res
                δ, ϕc_low, ϕc_hi =  find_critical_volume_fraction(Δ, Ns, key, datafolder_low_res)

                scatter!(ax, [(ϕc_low+ϕc_hi)/2], [δ], color=colmap[3])

            end

        elseif key == "A3"
            for Δ in σ_ratio_arr
                δ, ϕc_low, ϕc_hi =  find_critical_volume_fraction(Δ, Ns, key, datafolder)

                scatter!(ax, [(ϕc_low+ϕc_hi)/2], [δ],marker=:utriangle, color=colmap[4])
            end
            for Δ in σ_ratio_arr_low_res
                δ, ϕc_low, ϕc_hi =  find_critical_volume_fraction(Δ, Ns, key, datafolder_low_res)
                scatter!(ax, [(ϕc_low+ϕc_hi)/2], [δ], color=colmap[4])
            end

        end
    end
    x_pos_L, y_pos_L = 0.515, 0.4
    x_pos_G, y_pos_G = 0.525, 0.15
    text!(ax, x_pos_L, y_pos_L, text="Liquid")
    text!(ax, x_pos_G, y_pos_G, text="Ideal Glass")

    elem_1 = [MarkerElement(color = "black", marker =:circle, markersize = 15)]
    elem_2 = [MarkerElement(color = "black", marker =:utriangle, markersize = 15)]

    ha = -9.5
    va = 45.0
    Legend(fig[1:1,2], [elem_1, elem_2], [L"$N_k=100$", L"$N_k=300$"], 
            framevisible = false)

    display(fig)
    # save("Plots/phase_diagram_polydisperse_FINE_GRID.pdf", fig)
end
# plot_Phase_Diagram_fine_grid()

"""
    plot_avg_S()

At the critical point ϕ_c (first glassy state), plot the summed structure factor
S̄(k) = Σ_ij S_ij(k) and the species-averaged Lamb-Mössbauer factor f̄^(s)(k) for every fifth δ,
for the uniform (left) and inverse-cubic (right) distributions.
Saves `Plots/Structure_factor_PY_polydispersity.pdf` and `Plots/LM_PY_polydispersity.pdf`.
"""
function plot_avg_S()
    key_list = ["uniform", "gaussian", "A3"]

    num_phase_pts = 30
    Ns = 10

    Δ_arr_uniform = LinRange(0.1, 0.9, num_phase_pts)
    Δ_arr_gaussian = LinRange(0.05, 0.6, num_phase_pts)
    σ_ratio_arr = LinRange(1.0, 10, num_phase_pts)

    datafolder = "Data_Phase_Diagram"
    files = readdir(datafolder)

    colmap = cgrad(:matter, 31, categorical = true)

    fig = Figure(resolution=(1000,700))
    fig2 = Figure(resolution=(1000,700))

    fontsize_theme = Theme(fontsize=30)
    set_theme!(fontsize_theme)

    Makie.theme(:fonts).:regular = "CMU Serif"

    ax1 = Axis(fig[1,1], limits=(0.2, 39.8, 0, 4.0),
            ylabel=L"$\overline{S}(k)$", xlabel=L"$k$", 
            title=L"\text{Uniform}", xgridvisible=false, ygridvisible=false)
    ax2 = Axis(fig[1,2], limits=(0.2, 39.8, 0, 4.0), 
                ylabel=L"$\overline{S}(k)$", xlabel=L"$k$",
                title=L"\text{Inverse Cubic}",  xgridvisible=false, ygridvisible=false)

    ax1_b = Axis(fig2[1,1], limits=(0.5, 39.8, 0, 1.0),
                ylabel=L"$\overline{f^{(s)}}(k)$", xlabel=L"$k\overline{D}$", 
                title=L"\text{Uniform}",  xgridvisible=false, ygridvisible=false)
    ax2_b = Axis(fig2[1,2], limits=(0.5, 39.8, 0, 1.0), 
                    ylabel=L"$\overline{f^{(s)}}(k)$", xlabel=L"$k\overline{D}$",
                    title=L"\text{Inverse Cubic}",  xgridvisible=false, ygridvisible=false)
    

    key_list = ["uniform", "A3"]
    k_array = range(0.2, 39.8, length=100)

    for key in key_list

        if key == "uniform"
            count = 0
            for Δ in Δ_arr_uniform
                
                if count%5 == 0
                    δ, ϕc_low, ϕc_hi =  find_critical_volume_fraction(Δ, Ns, key, datafolder)

                    for file in files
                        if contains(file, string(ϕc_hi))
                            f = jldopen(joinpath(datafolder,file), "r")
                            S_avg = zeros(length(k_array))
                            f_avg = zeros(length(k_array))

                            for kid in 1:length(k_array)
                                val2 = 0.0
                                val3 = 0.0
                                for ns in 1:Ns 
                                    val3 += f["non_erg_param"]["f_c_$(ns)"][kid]

                                    for nss in 1:Ns 
                                        val2 += f["structure_factor"][kid][ns, nss]
                                    end
                                end
                                S_avg[kid] = val2
                                f_avg[kid] = val3
                            end

                            lines!(ax1, k_array, S_avg, 
                                linewidth=2.5,
                                color = colmap[count+3],
                                label=L"\delta = %$(round(δ,digits=2))")
                            lines!(ax1_b, k_array, f_avg./Ns, 
                                linewidth=2.5,
                                color = colmap[count+3],
                                label=L"\delta = %$(round(δ,digits=2))")

                            close(f)
                        end
                    end
                    count +=1
                else
                    count +=1 
                end

            end

            axislegend(ax1)
            axislegend(ax1_b)
        else
            count = 0
            for Δ in σ_ratio_arr
                
                if count%5 == 0
                    δ, ϕc_low, ϕc_hi =  find_critical_volume_fraction(Δ, Ns, key, datafolder)


                    for file in files
                        if contains(file, string(ϕc_hi))
                            f = jldopen(joinpath(datafolder,file), "r")
                            S_avg = zeros(length(k_array))
                            f_avg = zeros(length(k_array))

                            for kid in 1:length(k_array)
                                val2 = 0.0
                                val3 = 0.0
                                for ns in 1:Ns 
                                    val3 += f["non_erg_param"]["f_c_$(ns)"][kid]
                                    for nss in 1:Ns 
                                        val2 += f["structure_factor"][kid][ns, nss]
                                    end
                                end
                                S_avg[kid] =  val2
                                f_avg[kid] = val3
                            end

                            lines!(ax2, k_array, S_avg, 
                                linewidth=2.5,
                                color = colmap[count+3],
                                label=L"\delta = %$(round(δ,digits=2))")
                            lines!(ax2_b, k_array, f_avg./Ns, 
                                linewidth=2.5,
                                color = colmap[count+3],
                                label=L"\delta = %$(round(δ,digits=2))")

                                close(f)
                        end
                    end
                    count +=1
                else
                    count +=1 
                end

            end

            axislegend(ax2)
            axislegend(ax2_b)
        end

    end

    save("Plots/Structure_factor_PY_polydispersity.pdf", fig)
    save("Plots/LM_PY_polydispersity.pdf", fig2)

    display(fig)
    display(fig2)
end
# plot_avg_S()

"""
    plot_LM()

For each Δ of the inverse-cubic ("A3") scan, plot log10 f_1^(s)(k) of the smallest species (s = 1)
at every glassy volume fraction ϕ > ϕ_c, one figure per Δ.
"""
function plot_LM()
    #k_array = range(0.2, 39.8, length=200)
    
    kmax=40.0; Nk = 100; dk = kmax/Nk; k_array = dk*(collect(1:Nk) .- 0.5)
    Ns = 10

    datafolder = "Data_Phase_Diagram"

    files = readdir(datafolder)
    distrib_type = "A3"

    uniform_files = files[contains.(files, distrib_type)]
    Δ_arr_uniform = []

    for file in uniform_files
        Δ = parse(Float64, split(file, "_")[5])

        push!(Δ_arr_uniform, Δ)
    end

    Δ_arr_uniform = union(Δ_arr_uniform)

    for Δ in Δ_arr_uniform
        Δ_files = uniform_files[contains.(uniform_files, string(Δ)[2:end])]
        δ, ϕc_low, ϕc_hi =  find_critical_volume_fraction(Δ, Ns, distrib_type, datafolder)
        fname = uniform_files[contains.(uniform_files, "$(ϕc_hi)")][1]

        @show (δ, ϕc_hi)

        fontsize_theme = Theme(fontsize=30)
        set_theme!(fontsize_theme)
        Makie.theme(:fonts).:regular = "CMU Serif"
    
        fig2 = Figure(resolution=(1000,1000))

        ax_tl2 = Axis(fig2[1,1], ylabel=L"$f_{\sigma}^{(s)}(k)$"
                    , xticklabelsvisible=true, title=L"\delta = %$(round(δ, digits=4))",  xgridvisible=false, ygridvisible=false)#, xlabel=L"$(k\overline{D})^2$")


        for fname in Δ_files
            ϕ_file = parse(Float64, split(fname, "_")[7])
            f = jldopen(joinpath(datafolder, fname), "r")
            s=1

            D_arr = f["run_params"]["D_arr"]


            fs = f["non_erg_param"]["f_c_$(s)"]

            if ϕ_file > ϕc_hi && maximum(log10.(fs)) > -10
                @show (δ, round(D_arr[2]/D_arr[end],digits=3))
                lines!(ax_tl2, k_array, log10.(fs), 
                label = "$(ϕ_file)")
            end

            close(f)

        end
        axislegend(ax_tl2)
        display(fig2)
    end

end

# plot_LM()


"""
    plot_LM_log()

Plot log10 f_σ^(s)(k) vs (kD̄)² at ϕ_c for every species (and their average, in red), for the
uniform (left) and inverse-cubic (right) distributions at δ ≈ 0.07 (top) and δ ≈ 0.29 (bottom).
A straight line indicates a Gaussian f^(s)(k) ∼ exp(-(kD̄)²). Saves `Plots/log_LM.pdf`.
"""
function plot_LM_log()
    k_array = range(0.2, 39.8, length=100)
    Ns = 10
    # num_phase_pts = 30

    # Δ_arr_uniform = LinRange(0.1, 0.9, num_phase_pts)
    # Δ_arr_gaussian = LinRange(0.05, 0.6, num_phase_pts)
    # σ_ratio_arr = LinRange(1.0, 10, num_phase_pts)
    datafolder = "Data_Phase_Diagram"

    files = readdir(datafolder)

    fontsize_theme = Theme(fontsize=35)
    set_theme!(fontsize_theme)
    Makie.theme(:fonts).:regular = "CMU Serif"
    colmap = cgrad(:matter, 4, categorical = true)

    fig = Figure(resolution=(1000,1000))

    ax_tl = Axis(fig[1,1], ylabel=L"$\log\left(f_{\sigma}^{(s)}(k)\right)$"
                , xticklabelsvisible=false, title=L"\text{Uniform}",  xgridvisible=false, ygridvisible=false)#, xlabel=L"$(k\overline{D})^2$")
    ax_tr = Axis(fig[1,2], xticklabelsvisible=false, yticklabelsvisible=true, 
                title=L"\text{Inverse Cubic}",  xgridvisible=false, ygridvisible=false)#, ylabel=L"$\log\left(f_{\sigma}(k)\right)$", xlabel=L"$(k\overline{D})^2$")
    ax_bl = Axis(fig[2,1], ylabel=L"$\log\left(f_{\sigma}^{(s)}(k)\right)$", xlabel=L"$(k\overline{D})^2$",  xgridvisible=false, ygridvisible=false)
    ax_br = Axis(fig[2,2], xlabel=L"$(k\overline{D})^2$", 
                yticklabelsvisible=true,  xgridvisible=false, ygridvisible=false)


    # δ windows selecting the low (≈ 0.07) and high (≈ 0.29) polydispersity
    δmin_low = 0.06
    δmin_hi = 0.08


    δmax_low = 0.28
    δmax_hi = 0.30

    uniform_files = files[contains.(files, "uniform")]
    Δ_arr_uniform = []
    for file in uniform_files
        Δ = parse(Float64, split(file, "_")[5])

        push!(Δ_arr_uniform, Δ)
    end

    Δ_arr_uniform = union(Δ_arr_uniform)

    for Δ in Δ_arr_uniform
        δ, ϕc_low, ϕc_hi =  find_critical_volume_fraction(Δ, Ns, "uniform", datafolder)
        fname = uniform_files[contains.(uniform_files, "$(ϕc_hi)")][1]
        
        if δ >= δmin_low && δ <= δmin_hi
            f = jldopen(joinpath(datafolder, fname), "r")
            fs_avg = zeros(100)

            for s in 1:Ns
                fs = f["non_erg_param"]["f_c_$(s)"]
                fs_avg .+= fs
                lines!(ax_tl, k_array[5:55].^2, log10.(fs[5:55]), 
                    color=colmap[end])
            end

            lines!(ax_tl, k_array[5:55].^2, log10.(fs_avg[5:55]./Ns), 
            color="red")
            arrows!(ax_tl, [50], [-0.5], [0.0], [+0.5],
                    color="black", linewidth=2.0)
            text!(ax_tl, 55, -0.6, text=L"$D$")
            text!(ax_tl, 330, -0.08, text=L"$\delta = 0.07$")

            text!(ax_tl, 235, -0.45, text=L"$f_{\sigma}^{(s)}(k) \sim e^{-(k\overline{D}\,)^2}$", fontsize = 30)

            close(f)
        end
        if δ >= δmax_low && δ <= δmax_hi
            f = jldopen(joinpath(datafolder, fname), "r")
            fs_avg = zeros(100)

            for s in 1:Ns
                fs = f["non_erg_param"]["f_c_$(s)"]
                D_arr = f["run_params"]["D_arr"]
                fs_avg .+= fs
                lines!(ax_bl, k_array[5:55].^2, log10.(fs[5:55]), 
                    color=colmap[end])           
            end

            lines!(ax_bl, k_array[5:55].^2, log10.(fs_avg[5:55]./Ns), 
            color="red")

            text!(ax_bl, 330, -0.15, text=L"$\delta = 0.29$")

            close(f)
        end

    end

    A3_files = files[contains.(files, "A3")]
    A3_arr = []
    for file in A3_files
        Δ = parse(Float64, split(file, "_")[5])
        push!(A3_arr, Δ)
    end

    A3_arr = union(A3_arr)

    for Δ in A3_arr
        δ, ϕc_low, ϕc_hi =  find_critical_volume_fraction(Δ, Ns, "A3", datafolder)
        fname = A3_files[contains.(A3_files, "$(ϕc_hi)")][1]
        
        if δ >= δmin_low && δ <= δmin_hi
            f = jldopen(joinpath(datafolder, fname), "r")
            D_arr = f["run_params"]["D_arr"]
            fs_avg = zeros(100)
            for s in 1:Ns
                fs = f["non_erg_param"]["f_c_$(s)"]
                fs_avg .+= fs
                lines!(ax_tr, (k_array[5:55]).^2, log10.(fs[5:55]), 
                    color=colmap[end])
            end

            lines!(ax_tr, k_array[5:55].^2, log10.(fs_avg[5:55]./Ns), 
            color="red")

            text!(ax_tr, 330, -0.08, text=L"$\delta = 0.07$")

            close(f)
        end
        if δ >= δmax_low && δ <= δmax_hi
            f = jldopen(joinpath(datafolder, fname), "r")
            fs_avg = zeros(100)
            for s in 1:Ns
                fs = f["non_erg_param"]["f_c_$(s)"]
                fs_avg .+= fs
                lines!(ax_br, k_array[5:55].^2, log10.(fs[5:55]), 
                    color=colmap[end])
            end

            lines!(ax_br, k_array[5:55].^2, log10.(fs_avg[5:55]./Ns), 
            color="red", linewidth=1.4)
            text!(ax_br, 330, -0.15, text=L"$\delta = 0.29$")

            close(f)
        end

    end

    text!(ax_tl, 0, -1.5, text=L"\text{(a)}")
    text!(ax_bl, 0, -2.9, text=L"\text{(c)}")

    text!(ax_tr, 0, -1.5, text=L"\text{(b)}")
    text!(ax_br, 0, -2.9, text=L"\text{(d)}")

    display(fig)
    save("Plots/log_LM.pdf", fig)
end

# plot_LM_log()
