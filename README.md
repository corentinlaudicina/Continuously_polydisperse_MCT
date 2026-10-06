# Continuously polydisperse MCT

Mode-coupling theory (MCT) for polydisperse hard spheres with a continuous size distribution.
The distribution is discretised into `Ns` species, and the multicomponent MCT equations are
solved with Percus–Yevick structure factors using
[ModeCouplingTheory.jl](https://github.com/IlianPihlajamaa/ModeCouplingTheory.jl).

The scripts compute:

- the glass-transition (critical) volume fraction ϕ_c as a function of polydispersity;
- tagged-particle correlation functions F_s(k, t) approaching ϕ_c, or at fixed relaxation time;
- the corresponding susceptibilities χ''(ω), used to study the shape of the α-relaxation:
  KWW stretching, excess wings, and the MCT exponents γ, a and b.

## Requirements

Developed with Julia 1.7. Packages:

- Calculations: `ModeCouplingTheory`, `StaticArrays`, `QuadGK`, `Distributions`, `HDF5`, `JLD2`,
  `Roots`, `Dierckx`
- Analysis: also `CairoMakie`, `Plots`, `LsqFit`, `SpecialFunctions`, `Measurements`,
  `LaTeXStrings`, `DelimitedFiles`

The scripts call `Pkg.activate(".")`, so run them from the repository root after creating an
environment there:

```bash
julia --project=. -e 'using Pkg; Pkg.add(["ModeCouplingTheory", "StaticArrays", "QuadGK",
    "Distributions", "HDF5", "JLD2", "Roots", "Dierckx", "CairoMakie", "Plots", "LsqFit",
    "SpecialFunctions", "Measurements", "LaTeXStrings", "DelimitedFiles"])'
```

## Size distributions

Each distribution is discretised into `Ns` equimolar species, with diameters at the quantiles
(i - 1/2)/Ns of the distribution (`QuantizeDistribution.jl`). The mean diameter is 1, and the
polydispersity δ is the standard deviation of the diameters.

| `key`      | Distribution                       | Width parameter `Δ`  |
|------------|------------------------------------|----------------------|
| `uniform`  | uniform on [1 - Δ, 1 + Δ]          | half-width           |
| `gaussian` | Gaussian                           | standard deviation   |
| `A3`       | P(σ) ∝ σ⁻³ ("inverse cubic")        | size ratio           |

## Workflow

The calculations are run as SLURM jobs. The `run_all_*.jl` scripts submit them with `sbatch`
through small wrapper scripts (`*.run`, not included here) that load Julia and run
`julia <script>.jl "$@"`. Output folders must exist before running.

1. **Critical points.** `julia calculate_critical_point.jl Ns Δ μ key`
   finds ϕ_c by bisection on the non-ergodicity parameter, and writes every evaluation to
   `Data_Phase_Diagram_Fine_Grid/fc_Ns_<Ns>_poly_<Δ>_phi_<ϕ>_<key>.jld2`.
   `run_all_calculate_critical_point.jl` scans Ns and Δ.
2. **Dynamics approaching ϕ_c.** `julia calculate_MCT_solution.jl Ns Δ ϕ key` (`key` = `A3` or
   `gaussian`) solves the collective and tagged MCT equations and writes
   `Data_sol_<key>_epsilon/mctsol_Ns_<Ns>_Delta_<Δ>_phi_<ϕ>.hdf5`.
   `run_all_MCT_solutions.jl` reads ϕ_c from `Data_Phase_Diagram` and submits
   ϕ = ϕ_c(1 - ϵ) for ϵ from 10⁻¹ to 10⁻⁵.
3. **Fixed relaxation time.** `julia calculate_const_relax_time.jl Δ` finds the volume fraction
   where τ_α = 10¹⁰ and writes `Data_sol/mctsol_Ns_<Ns>_Delta_<Δ>.hdf5`. Progress is logged in
   `Errors/`. `run_all_calculate_const_relax_time.jl` scans Δ.

### Output files

MCT solutions (HDF5):

- `run_params`: `Ns`, `delta`, `phi`, `D_arr`, `distrib_type`, `k_peak`
- `F_tagged`: `t_array` and `F_s_<s>`, the tagged correlation function of species `s` at the
  peak of the summed structure factor
- `relaxation_spectrum`: `omega_arr` and `chi_s_<s>`

Critical points (JLD2):

- `run_params`: `Ns`, `D_arr`, `Delta`, `delta_var`, `distribution_type`
- `non_erg_param`: `fc_collective` and `f_c_<s>`
- `structure_factor`

## Repository layout

| File                                | Contents                                                       |
|-------------------------------------|----------------------------------------------------------------|
| `PercusYevick.jl`                   | PY direct correlation function and structure factor of a mixture |
| `QuantizeDistribution.jl`           | discretisation of the size distributions                       |
| `calculate_critical_point.jl`       | critical volume fraction from the non-ergodicity parameter     |
| `calculate_MCT_solution.jl`         | MCT dynamics and χ''(ω) at a given ϕ                           |
| `calculate_const_relax_time.jl`     | MCT dynamics at fixed τ_α                                      |
| `run_all_*.jl`                      | SLURM job submission                                           |
| `Analysis/`                         | analysis and figure scripts                                    |

In `Analysis/`:

| File                                | Contents                                                       |
|-------------------------------------|----------------------------------------------------------------|
| `plot_phase_diagram.jl`             | phase diagrams, S(k) and Lamb–Mössbauer factors at ϕ_c          |
| `plot_excess_wings.jl`              | χ''(ω), excess wings, KWW exponents, MCT exponents vs δ        |
| `convergence_structure_factor.jl`   | convergence with the number of species                         |
| `find_critical_exponents.jl`        | fits of γ, a and b                                             |
| `find_stretched_exp_params.jl`      | KWW fits of the α-relaxation                                   |
| `plot_KWW_exponents.jl`             | KWW exponents vs δ and vs particle size                        |
| `plot_relaxation_time.jl`           | F̄_s(t) and χ̄''(ω) for a set of solutions                      |
| `calculate_susceptibility.jl`       | χ''(ω) from F(t), and recomputing it for stored solutions      |
| `find_relaxation_time.jl`           | helper functions shared by the analysis scripts                |

Run the analysis scripts from the repository root, e.g. `julia Analysis/plot_excess_wings.jl`.
They are research scripts: data folders are hard-coded, and some read earlier versions of the
output format.

## Susceptibility

χ''(ω) is computed from F(t) as

χ''(ω) = ∫ G(u) ωe^u / (1 + ω²e^{2u}) du,  with u = ln t and G(u) = -dF/du,

i.e. -dF/d ln t is treated as the distribution of relaxation times.

## Reference

Percus–Yevick solution for mixtures: R. J. Baxter, J. Chem. Phys. 52, 4559 (1970).
