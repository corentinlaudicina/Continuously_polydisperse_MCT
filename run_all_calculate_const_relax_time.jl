# Submit (via SLURM `sbatch`) one constant-relaxation-time calculation per polydispersity Δ
# (see calculate_const_relax_time.jl).

Δ_arr = LinRange(0.1, 0.9, 11)

for Δ in Δ_arr
    mycommand = Cmd(`sbatch calculate_const_relax_time.run $Δ`)
    run(mycommand)
end
