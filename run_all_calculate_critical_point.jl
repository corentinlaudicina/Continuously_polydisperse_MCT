# Submit (via SLURM `sbatch`) one critical-point calculation per (Ns, Δ) pair
# (see calculate_critical_point.jl), scanning the width parameter of the size distribution `key`.

Ns_arr = [2,5,10]
num_phase_pts = 15

keys_list = ["uniform"]#, "gaussian", "A3"]

Δ_arr_uniform = LinRange(0.1, 0.9, num_phase_pts)
Δ_arr_gaussian = LinRange(0.05, 0.6, num_phase_pts)
σ_ratio_arr = LinRange(1.0, 10, num_phase_pts)

for key in keys_list

    if key == "uniform"

        μ_uniform = 1.0

        for Ns in Ns_arr
            for Δ in Δ_arr_uniform
                mycommand = Cmd(`sbatch run_calculate_critical_point.run $Ns $Δ $μ_uniform $key`)
                run(mycommand)
            end
        end

    elseif key == "gaussian"

        μ_gaussian = 1.0

        for Ns in Ns_arr
            for Δ in Δ_arr_gaussian
                mycommand = Cmd(`sbatch run_calculate_critical_point.run $Ns $Δ $μ_gaussian $key`)
                run(mycommand)
            end
        end

    elseif key == "A3"

        μ = 1.0

        for Ns in Ns_arr
            for σ_ratio in σ_ratio_arr
                mycommand = Cmd(`sbatch run_calculate_critical_point.run $Ns $σ_ratio $μ $key`)
                run(mycommand)
            end
        end
    else
        println("key undefined")
        continue
    end
end
