#=
Submit (via SLURM `sbatch`) one MCT calculation per state point approaching the glass
transition from the fluid side, for a series of polydispersities.

For each polydispersity Δ, the critical volume fraction ϕ_c is read from the non-ergodicity
parameters previously computed in `Data_Phase_Diagram` (midpoint of the bracketing interval),
and one job `run_calculate_MCT_solution_unix.run` is submitted for every
ϕ = ϕ_c (1 - ϵ), ϵ = 10⁻¹ ... 10⁻⁵ (15 log-spaced values).
=#

import Pkg; Pkg.activate(".")

using Statistics, JLD2
using StaticArrays

"""
    find_critical_volume_fraction(Δ, Ns, key, datafolder)

Bracket the critical volume fraction for width parameter `Δ`, `Ns` species and distribution `key`
using the `fc_*.jld2` files in `datafolder`. The mixture is considered glassy when
the sum of the collective non-ergodicity parameter exceeds 10⁻⁴.

Returns `(δ, ϕ_low, ϕ_hi)`: the standard deviation of the diameters, the largest volume fraction
without a glass solution and the smallest one with it.
"""
function find_critical_volume_fraction(Δ, Ns, key, datafolder)

    files = readdir(datafolder)

    files = files[contains.(files, "Ns_$(Ns)_")]
    files = files[contains.(files, key)]
    files = files[contains.(files, "poly_$(Δ)_")]

    mydict = Dict{Float64, Float64}()

    δ = 0
    count = 0
    for file in files
        f = jldopen(joinpath(datafolder, file), "r")
        fc = f["non_erg_param"]["fc_collective"]
        ϕ = parse(Float64, split(file, '_')[7])
        mydict[ϕ] = sum(sum(fc))
        if count == 0
            δ += std(f["run_params"]["D_arr"] ; corrected=false)
            count +=1
        end
        close(f)
    end
    sorted_ϕ_arr = sort(collect(keys(mydict)))

    for (i, ϕ) in enumerate(sorted_ϕ_arr)
        fc = mydict[ϕ]
        if fc > 0.0001
            return δ, sorted_ϕ_arr[i-1], sorted_ϕ_arr[i]
        end
    end
end

num_phase_pts_low_res = 30

Δ_arr_gaussian = LinRange(0.05, 0.6, num_phase_pts_low_res)
σ_ratio_arr = LinRange(1.0, 10, num_phase_pts_low_res)

dir = "Data_Phase_Diagram"

Ns = 10

keys_list = ["A3"]

# relative distances ϵ to the critical point: ϕ = ϕ_c (1 - ϵ)
ϵ_arr = 10.0.^LinRange(-1,-5, 15)

for key in keys_list

    if key == "A3"
        @show key
        for Δ_id in 2:2:length(σ_ratio_arr)-5
            Δ = σ_ratio_arr[Δ_id]
            δ, ϕlow, ϕhi = find_critical_volume_fraction(Δ, Ns, key, dir)
            ϕc = (ϕlow +ϕhi)/2

            @show δ, ϕc
            for ϵ in ϵ_arr
                ϕ = ϕc*(1-ϵ)

                mycommand = Cmd(`sbatch run_calculate_MCT_solution_unix.run $Ns $Δ $ϕ $key`)
                run(mycommand)
            end
        end

    elseif key == "gaussian"
        @show key
        for Δ_id in 2:2:length(Δ_arr_gaussian)-4
            Δ = Δ_arr_gaussian[Δ_id]
            δ, ϕlow, ϕhi = find_critical_volume_fraction(Δ, Ns, key, dir)
            ϕc = (ϕlow +ϕhi)/2
            @show δ, ϕc

            for ϵ in ϵ_arr
                ϕ = ϕc*(1-ϵ)

                mycommand = Cmd(`sbatch run_calculate_MCT_solution_unix.run $Ns $Δ $ϕ $key`)
                run(mycommand)
            end
        end
    end
end
