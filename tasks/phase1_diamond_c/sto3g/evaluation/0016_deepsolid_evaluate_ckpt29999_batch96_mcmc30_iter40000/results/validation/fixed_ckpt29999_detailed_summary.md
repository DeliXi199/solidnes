# Fixed Checkpoint 29999 Evaluation Summary

Experiment: `diamond_c_deepsolid_evaluate_ckpt29999_batch96_mcmc30_iter40000`
Job: `120884`, `COMPLETED 0:0`, elapsed `00:15:48` on `gpu40904`.

## Setup

- Restored checkpoint: `qmcjax_ckpt_029999.npz` from the 30000-step training trajectory.
- Optimizer: `none`.
- Batch size: `96`.
- MCMC steps per iteration: `30`.
- Evaluation rows: `10000`, steps `30000` through `39999`.
- Runtime resources: `6 x RTX 4090`, `96 CPU`.
- HF reference: `-74.004196731601 Ha`.

## Energy Estimate

- all: mean `-74.021768218405 Ha`, gap to HF `-0.017571486804 Ha`, block stderr `0.006933 Ha`, pmove `0.5377`.
- first_half: mean `-74.017387225155 Ha`, gap to HF `-0.013190493555 Ha`, block stderr `0.009015 Ha`, pmove `0.5377`.
- second_half: mean `-74.026149211655 Ha`, gap to HF `-0.021952480054 Ha`, block stderr `0.011863 Ha`, pmove `0.5377`.
- last_5000: mean `-74.026149211655 Ha`, gap to HF `-0.021952480054 Ha`, block stderr `0.011863 Ha`, pmove `0.5377`.
- last_2000: mean `-74.041329701214 Ha`, gap to HF `-0.037132969613 Ha`, block stderr `0.021497 Ha`, pmove `0.5374`.
- last_1000: mean `-74.030545544440 Ha`, gap to HF `-0.026348812839 Ha`, block stderr `0.035332 Ha`, pmove `0.5373`.
- last_500: mean `-74.032623736867 Ha`, gap to HF `-0.028427005267 Ha`, block stderr `0.038764 Ha`, pmove `0.5364`.

## Block Means

- Block 01 steps 30000-30999: energy `-74.027360247255 Ha`, gap `-0.023163515654 Ha`, variance `23.2566`, pmove `0.5379`.
- Block 02 steps 31000-31999: energy `-73.985915675160 Ha`, gap `+0.018281056441 Ha`, variance `23.4460`, pmove `0.5395`.
- Block 03 steps 32000-32999: energy `-74.040995741007 Ha`, gap `-0.036799009406 Ha`, variance `26.1594`, pmove `0.5363`.
- Block 04 steps 33000-33999: energy `-73.995627263581 Ha`, gap `+0.008569468020 Ha`, variance `23.6743`, pmove `0.5376`.
- Block 05 steps 34000-34999: energy `-74.037037198775 Ha`, gap `-0.032840467174 Ha`, variance `23.9158`, pmove `0.5373`.
- Block 06 steps 35000-35999: energy `-74.036458466466 Ha`, gap `-0.032261734866 Ha`, variance `25.1855`, pmove `0.5370`.
- Block 07 steps 36000-36999: energy `-74.006170248309 Ha`, gap `-0.001973516708 Ha`, variance `24.5094`, pmove `0.5388`.
- Block 08 steps 37000-37999: energy `-74.005457941069 Ha`, gap `-0.001261209468 Ha`, variance `25.5116`, pmove `0.5376`.
- Block 09 steps 38000-38999: energy `-74.052113857988 Ha`, gap `-0.047917126387 Ha`, variance `24.6788`, pmove `0.5376`.
- Block 10 steps 39000-39999: energy `-74.030545544440 Ha`, gap `-0.026348812839 Ha`, variance `23.5600`, pmove `0.5373`.

## Interpretation

The independent fixed-checkpoint estimate confirms the checkpoint is near the same-cell HF reference. The all-sample mean is about 0.018 Ha below the HF reference with a 10-block standard error of about 0.007 Ha. Since this is a same-setup HF comparison rather than an exact physical benchmark, this should be treated as a validation success for the optimization/sampling workflow, followed by either a repeat fixed evaluation or a physics upgrade such as basis/supercell.
