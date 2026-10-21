# GAPS-US

**Graph-Based Active Path Sampling for Adaptive Umbrella Sampling of Multidimensional Free-Energy Landscapes**

Research code associated with the manuscript in preparation by Mohsen Farshad and Gaurav Arya (Duke University).

## Overview

GAPS-US extends graph-based active path sampling (GAPS) to **adaptive umbrella sampling**. It uses:

- **Weighted Histogram Analysis Method (WHAM)** to reconstruct free-energy landscapes from biased simulations.
- **Gaussian-process regression and active learning** to select informative new umbrella windows.
- A **measured-overlap graph** to determine whether newly sampled windows are thermodynamically connected.
- **Bridge umbrellas** when additional physical sampling is required to connect a selected target.

All information-selected and bridge umbrellas contribute to the physical sampling cost.

## Benchmark systems

1. **Müller–Brown potential** — a two-dimensional analytical free-energy landscape.
2. **Alanine dipeptide** — molecular umbrella sampling in the periodic dihedral coordinates `(phi, psi)`.
3. **Lennard-Jones + Axilrod–Teller–Muto (LJ+ATM) trimer** — a constrained three-dimensional distance-based landscape with permutation symmetry and nonadditive three-body interactions.

## Source code

- `unified_GAPS_US-v4.py` — unified research launcher and system-specific implementations for the three benchmarks.

The launcher includes user-configurable `SYSTEM` and `START_NEW_CAMPAIGN` settings near the top. **Inspect these settings before execution**: the source may be configured to start a new simulation campaign. Paths and external MD dependencies are environment-specific.

## Manuscript plotting scripts

| Script | Manuscript figure(s) |
| --- | --- |
| `plotting/plot_Figures_2_3.py` | Figures 2 and 3: Müller–Brown results |
| `plotting/plot_Figure_4.py` | Figure 4: Alanine dipeptide |
| `plotting/plot_Figures_5_6.py` | Figures 5 and 6: LJ+ATM trimer |
| `plotting/plot_Figure_7.py` | Figure 7: LJ+ATM pair/nonadditive decomposition |

Figure 1 is a conceptual workflow illustration, rather than an output of these plotting scripts.

## Reproducibility notes

The plotting programs read saved campaign results; they are **not** a substitute for the underlying simulation inputs and trajectory data. The complete simulation environment and campaign configuration may require GROMACS and/or LAMMPS, additional Python packages, and access to benchmark-specific inputs. Some scripts contain CRC-specific absolute paths that must be updated for use elsewhere.

Large trajectories and archived campaign outputs are intentionally not tracked in this source-code repository. Numerical results in the manuscript remain subject to final scientific validation.

## Manuscript status

**Manuscript in preparation.** The draft is not included in this public repository. Supporting Information and validated data may be added separately after verification.

## Citation

A formal citation and DOI will be added when the manuscript is available.
