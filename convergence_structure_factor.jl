#=
Convergence of the results with the number of species n (= Ns) used to discretise a uniform
size distribution, at fixed polydispersity δ = std(D).

    convergence_structure_factor()    PY summed structure factor S̄(k) and its peak height
                                      vs n, for δ = 0.2 and δ = 0.4 at ϕ = 0.521
    plot_convergence_Fk()             species-averaged tagged correlation function vs n,
                                      for δ = 0.4 → Plots/Fk_convergence_uniform.pdf
    plot_Phase_Diagram_convergence()  critical volume fraction ϕ_c vs δ for n = 2, 5, 10
                                      → Plots/phase_diagram_convergence.pdf

Only `plot_convergence_Fk()` is called when the file is run.
=#

using CairoMakie, Roots, HDF5

include("../QuantizeDistribution.jl")
include("../PercusYevick.jl")

"""
    find_d(N, Δ, μ, key)

Return the `N` species diameters for the size distribution `key` ("uniform", "gaussian" or "A3").
`μ` is the mean diameter and `Δ` the width parameter of the distribution
(for "A3", `Δ` acts as the size ratio and `μ` is unused).
"""
function find_d(N, Δ, μ, key)
    if (key == "uniform")

        d = find_uniform_quantization(μ, Δ, N)
        # @show d
        return d 
    elseif (key == "gaussian")
        d = find_gaussian_quantization(μ,Δ,N)
        # @show d
        return d 
    elseif (key == "A3")
        ## BEWARE FOR THIS Δ ACTS AS SIZE RATIO
        d = sample_A_sigma3(Δ, N)
        # @show d 
        return d
    else
        error("DISTRIBUTION UNDEFINED")
    end
end

"""
    find_critical_volume_fraction(Δ, Ns, key, datafolder)

Bracket the critical volume fraction for width parameter `Δ`, `Ns` species and distribution `key`
using the `fc_Ns_<Ns>_poly_<Δ>_phi_<ϕ>_<key>.jld2` files in `datafolder`. A state is
considered glassy when the sum of the collective non-ergodicity parameter exceeds 10⁻⁴.

Returns `(δ, ϕ_low, ϕ_hi)`: the standard deviation of the diameters, the largest volume fraction
without a glass solution and the smallest one with it.
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
                δ += std(f["run_params"]["D_arr"] ; corrected=false)
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
    average_Sk(Sk, Ns)

Return the summed structure factor S̄(k) = Σ_ij S_ij(k) at each wave vector, from a vector of
`Ns × Ns` partial structure factor matrices.
"""
function average_Sk(Sk, Ns)

    Nk = length(Sk)
    Sk_avg = zeros(Nk)

    for n1 in 1:Ns 
        for n2 in 1:Ns 
            for ik in 1:Nk
                Sk_avg[ik] += Sk[ik][n1,n2]
            end
        end
    end
    return Sk_avg
end


"""
    objective(target_δ, Δ, μ, Ns, key)

Root-finding objective `target_δ - δ(Δ)`, where δ is the standard deviation of the `Ns`
discretised diameters for width parameter `Δ`.
"""
function objective(target_δ, Δ, μ, Ns, key)
    d = find_d(Ns, Δ, μ, key)
    δ = std(d ; corrected=false)
    return target_δ - δ
end

"""
    find_target_δ(target_δ, μ, Ns, key)

Find by bisection on Δ ∈ (10⁻⁴, 1) the width parameter for which the `Ns` discretised diameters
have standard deviation `target_δ`. Used to compare different `Ns` at the same polydispersity.
"""
function find_target_δ(target_δ, μ, Ns, key)
    f = Δ -> objective(target_δ, Δ, μ, Ns, key)
    Δ = find_zero(f, (0.0001, 1.0), Roots.Bisection(), atol=10.0^-8, verbose=true)
    return Δ
end

# Ns_arr = [2,4,6,8,10,15,20]

# target_δ = 0.4
# key = "uniform"
# μ = 1.0 

# Δ_arr = LinRange(0.05, 0.8, 5)
# for Δ in Δ_arr
#     d = find_d(10, Δ, 1.0, key)
#     δ = std(d ; corrected=false)
#     @show δ
# end
# Δ_A3_arr_low = Float64[]
# Δ_A3_arr_hi = Float64[]

# for Ns in Ns_arr
#     Δ = find_target_δ(target_δ, μ, Ns, key)
#     d = find_d(10, Δ, 1.0, key)
#     δ = std(d ; corrected=false)
#     @show Δ, δ

#     push!(Δ_A3_arr_hi, Δ)
# end

# @show Δ_A3_arr_hi
# @show Δ_A3_arr_hi

"""
    convergence_structure_factor()

Plot the PY summed structure factor S̄(k) (bottom) and its peak height S̄(k*) (top) as a function of
the number of species n, for a uniform distribution with δ = 0.2 (left) and δ = 0.4 (right) at ϕ = 0.521.
"""
function convergence_structure_factor()
    # Δ giving δ = 0.2 (low) and δ = 0.4 (hi) for each n in Ns_arr (obtained with find_target_δ)
    Δ_uniform_arr_low =[0.40000000935047847, 0.35777089096605763, 0.3513240370899437, 0.34914863158017384, 0.34815531190484744, 0.3471825429797171, 0.3468439735472201]

    Δ_uniform_arr_hi = [0.8000000048801299, 0.7155417681112882, 0.7026480603590604, 0.6982972493395204, 0.6963106099888677, 0.694365072138607, 0.6936879666894669]

    # Δ_A3_arr_low = [3.023022472858429, 2.130876123905182, 2.056818261742592, 2.0334149599075317, 2.022958993911743, 2.0128493905067444, 2.0093598067760468]

    # Δ_A3_arr_hi = [3.023022472858429, 2.130876123905182, 2.056818261742592, 2.0334149599075317, 2.022958993911743, 2.0128493905067444, 2.0093598067760468]


    kmax = 40.0
    Nk = 400

    ϕ =  0.521

    μ = 1.0
    key = "uniform"

    Ns_arr = [2,4,6,8,10,15,20]

    colmap = cgrad(:roma, length(Ns_arr)+1, categorical = true)

    fontsize_theme = Theme(fontsize=30)
    set_theme!(fontsize_theme)

    Makie.theme(:fonts).:regular = "CMU Serif"

    fig = Figure(resolution=(900,750))

    ax1_t = Axis(fig[1,1], title=L"$\delta = 0.2$",
                    xlabel=L"$n$", 
                    ylabel=L"$\overline{S}(k^{*})$",
                    xticklabelsize=20, yticklabelsize=20, xlabelpadding=-10,
                    xgridvisible=false, ygridvisible=false
                )
    ax2_t = Axis(fig[1,2], title=L"$\delta = 0.4$",
                xlabel=L"$n$", 
                xticklabelsize=20, yticklabelsize=20, xlabelpadding=-10,
                xgridvisible=false, ygridvisible=false 
                # ylabel=L"$\overline{S}(k^{*})$",
        )
        
    # ax1_t.aspect = 2
    # ax2_t.aspect = 2

    ax1 = Axis(fig[2,1], xlabel=L"$k$", ylabel=L"$\overline{S}(k)$",xticklabelsize=20, yticklabelsize=20, height=400,  xgridvisible=false, ygridvisible=false)
    ax2 = Axis(fig[2,2], xlabel=L"$k$", height=400,xticklabelsize=20, yticklabelsize=20,  xgridvisible=false, ygridvisible=false)

    Ns_count_label = 1
    elems = [LineElement(color = colmap[Ns_count_label+1]) for Ns_count_label in 1:length(Ns_arr)]

    Legend(fig[1:2,3], [i for i in elems], [L"$n=%$(Ns)" for Ns in Ns_arr], 
            framevisible = false)


    xlims!(ax1, 0,20)
    xlims!(ax2, 0,20)

    rowgap!(fig.layout, 20)

    ff = Figure(resolution=(500,500))
    axx = Axis(ff[1,1])

    count = 2
    count_Ns = 1
    for Ns in Ns_arr
        Δ = Δ_uniform_arr_low[count_Ns]
        d = find_d(Ns, Δ, μ, key)
        δ = std(d ; corrected=false)
        @show δ
        x = [1/Ns for _ in 1:Ns]
        ρ_all = 6ϕ/(π*sum(x .* d .^3))
        ρ = x*ρ_all

        dk = kmax/Nk
        k_array = dk*(collect(1:Nk) .- 0.5)

        Sk = [SMatrix{Ns,Ns}(find_structure_factor_PY(k, d, ρ)) for k in k_array]
        
        Sk_avg = average_Sk(Sk, Ns)
        
        max_Sk = maximum(Sk_avg)
        lines!(ax1, k_array, Sk_avg, color=colmap[count], label=false)
        scatter!(ax1_t, [Ns], [max_Sk], color="black", label=false)
        count += 1
        count_Ns += 1
    end

    count = 2
    count_Ns = 1
    for Ns in Ns_arr
        Δ = Δ_uniform_arr_hi[count_Ns]
        d = find_d(Ns, Δ, μ, key)
        δ = std(d ; corrected=false)
        @show δ
        x = [1/Ns for _ in 1:Ns]
        ρ_all = 6ϕ/(π*sum(x .* d .^3))
        ρ = x*ρ_all

        dk = kmax/Nk
        k_array = dk*(collect(1:Nk) .- 0.5)

        Sk = [SMatrix{Ns,Ns}(find_structure_factor_PY(k, d, ρ)) for k in k_array]
        
        Sk_avg = average_Sk(Sk, Ns)
        
        max_Sk = maximum(Sk_avg)

        lines!(ax2, k_array, Sk_avg, color=colmap[count])
        lines!(axx, k_array, Sk_avg, color=colmap[count])

        scatter!(ax2_t, [Ns], [max_Sk], color="black", label=false)
        count += 1
        count_Ns += 1
    end
    # display(ff)

    text!(ax1_t, 18.5, 1.9783, text=L"\text{(a)}", fontsize=20)
    text!(ax2_t, 18.5, 1.465, text=L"\text{(c)}", fontsize=20)

    text!(ax1, 17.5, 1.930, text=L"\text{(b)}", fontsize=20)
    text!(ax2, 17.5, 1.455, text=L"\text{(d)}", fontsize=20)

    display(fig)

    # save("Plots/Sk_convergence_uniform.pdf", fig)
end

# convergence_structure_factor()

"""
    plot_convergence_Fk()

Plot the species-averaged tagged correlation function F̄_s(k, t) vs log t for n = 2 to 10 species
at δ = 0.4, from the MCT solutions in `Data_fixed_poly`. The n = 10 curve is dashed.
Saves `Plots/Fk_convergence_uniform.pdf`.
"""
function plot_convergence_Fk()
    Ns_arr = [2,4,6,8,10]#,15,20]

    colmap = cgrad(:roma, length(Ns_arr)+1, categorical = true)

    fontsize_theme = Theme(fontsize=25)
    set_theme!(fontsize_theme)

    Makie.theme(:fonts).:regular = "CMU Serif"

    fig = Figure(resolution=(800,350))

    ax1 = Axis(fig[1,1], xlabel=L"$\log(t)$", ylabel=L"$\overline{F^{(s)}}(k,\, t)$",xticklabelsize=20, yticklabelsize=20, yminorticksvisible = true, limits=(-2,7,0,1),  xgridvisible=false, ygridvisible=false)
    # ax2 = Axis(fig[1,2], xlabel=L"$t$",xticklabelsize=20, yticklabelsize=20, limits=(-2,6,0,1))

    Ns_count_label = 1
    elems = [LineElement(color = colmap[Ns_count_label+1]) for Ns_count_label in 1:length(Ns_arr)-1]

    push!(elems, LineElement(color = colmap[end], linestyle=:dash))

    Legend(fig[1:1,2], [i for i in elems[1:end]], [L"$n=%$(Ns)" for Ns in Ns_arr[1:end]], 
            framevisible = false)

    Δ_uniform_arr_low_519 = [0.40000000935047847, 0.35777089096605763, 0.3513240370899437, 0.34914863158017384, 0.34815531190484744]

    Δ_uniform_arr_hi_519 = [0.8000000048801299, 0.7155417681112882, 0.7026480603590604, 0.6982972493395204, 0.6963106099888677]
        

    dir = "Data_fixed_poly"

    fnames = readdir(dir)

    fnames_low = String[]
    fnames_hi = String[]

    # split the files into δ = 0.2 (low) and δ = 0.4 (hi) runs
    for fname in fnames

        Δ = parse(Float64, split(fname, "_")[5])
        ϕ = parse(Float64, split(fname, "_")[end][1:end-5])
        if Δ in Δ_uniform_arr_hi_519
            push!(fnames_hi, fname)
        else
            push!(fnames_low, fname)
        end
    end

    sort!(fnames_hi)
    sort!(fnames_low)

    # alphabetical order puts Ns_10 first: move it to the end so that the colours follow n
    push!(fnames_hi, fnames_hi[1])
    deleteat!(fnames_hi,1)

    # count = 1
    # for fname in fnames_low
    #     f = h5open(joinpath(dir, fname), "r")
    #     Ns = read(f["run_params"], "Ns")

    #     t_array = read(f["F_tagged"], "t_array")
    #     Fs_avg = zeros(length(t_array))

    #     for s in 1:Ns 
    #         Fs = read(f["F_tagged"], "F_s_$(s)")
    #         Fs_avg .+= Fs
    #     end

    #     Fs_avg ./= Ns
    #     lines!(ax1, log10.(t_array[2:end]), Fs_avg[2:end], color=colmap[count])
    #     count += 1
    # end

    count = 2
    for fname in fnames_hi[1:end]
        f = h5open(joinpath(dir, fname), "r")
        Ns = read(f["run_params"], "Ns")
        @show fname
        t_array = read(f["F_tagged"], "t_array")
        Fs_avg = zeros(length(t_array))

        for s in 1:Ns 
            Fs = read(f["F_tagged"], "F_s_$(s)")
            Fs_avg .+= Fs
        end

        Fs_avg ./= Ns
        if Ns == 10
            lines!(ax1, log10.(t_array[2:end]), Fs_avg[2:end], color=colmap[count], linestyle=:dash)
        else
            lines!(ax1, log10.(t_array[2:end]), Fs_avg[2:end], color=colmap[count])
        end
        count += 1
    end

    text!(ax1, 6, 0.9, text=L"\delta=0.4", fontsize=20)

    display(fig)
    save("Plots/Fk_convergence_uniform.pdf", fig)
end

plot_convergence_Fk()

"""
    plot_Phase_Diagram_convergence()

Plot the critical volume fraction ϕ_c vs polydispersity δ for a uniform distribution discretised
into n = 2, 5 and 10 species (data in `Data_Phase_Diagram_Fine_Grid`).
Saves `Plots/phase_diagram_convergence.pdf`.
"""
function plot_Phase_Diagram_convergence()
    key = "uniform"

    num_phase_pts = 15 #30 for low resolution
    Ns_arr = [2,5,10]

    num_phase_pts_low_res = 15 

    Δ_arr_uniform = LinRange(0.1, 0.9, num_phase_pts)

    datafolder = "Data_Phase_Diagram_Fine_Grid"

    colmap = cgrad(:roma, 4, categorical = true)

    fig = Figure(resolution=(700,500))

    fontsize_theme = Theme(fontsize=30)
    set_theme!(fontsize_theme)

    Makie.theme(:fonts).:regular = "CMU Serif"

    ax = Axis(fig[1,1], 
                xlabel=L"$\varphi_c$", 
                ylabel=L"$\delta$", 
                limits=(0.513, 0.525, 0.05, 0.49), 
                xgridvisible=false, ygridvisible=false)

                count = 2
        for Ns in Ns_arr
            for Δ in Δ_arr_uniform
                δ, ϕc_low, ϕc_hi =  find_critical_volume_fraction(Δ, Ns, key, datafolder)
                scatter!(ax, [(ϕc_low+ϕc_hi)/2], [δ], color=colmap[count], 
                label = false)
            end
            count +=1
        end
    # x_pos_L, y_pos_L = 0.515, 0.4
    # x_pos_G, y_pos_G = 0.525, 0.15
    # text!(ax, x_pos_L, y_pos_L, text="Liquid")
    # text!(ax, x_pos_G, y_pos_G, text="Ideal Glass")

    elem_1 = [MarkerElement(color = colmap[2], marker =:circle, markersize = 15)]
    elem_2 = [MarkerElement(color = colmap[3], marker =:circle, markersize = 15)]
    elem_3 = [MarkerElement(color = colmap[4], marker =:circle, markersize = 15)]

    ha = 0.5
    va = 0.3

    Legend(fig[1:1,2], [elem_1, elem_2, elem_3], [L"$n=2$", L"$n=5$", L"$n=10$"], 
            framevisible = false)

    display(fig)
    save("Plots/phase_diagram_convergence.pdf", fig)
end
# plot_Phase_Diagram_convergence()