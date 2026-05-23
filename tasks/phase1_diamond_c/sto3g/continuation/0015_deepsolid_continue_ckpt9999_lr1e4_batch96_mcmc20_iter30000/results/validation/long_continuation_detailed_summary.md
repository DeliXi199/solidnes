# 20000-Step LR1e-4 Continuation Summary

Experiment: `diamond_c_deepsolid_continue_ckpt9999_lr1e4_batch96_mcmc20_iter30000`
Job: `120805`, `COMPLETED 0:0`, elapsed `00:27:36` on `gpu40904`.

## Setup

- Restored checkpoint: `qmcjax_ckpt_009999.npz` from the lr2e-4 continuation.
- Optimizer: `adam`.
- Learning rate: `1e-4`, delay `10000`, decay `1.0`.
- Batch size: `96`.
- MCMC steps per iteration: `20`.
- Continuation rows: `20000`, steps `10000` through `29999`.
- Runtime resources: `6 x RTX 4090`, `96 CPU`.
- HF reference: `-74.004196731601 Ha`.

## Energy Estimate

- all: mean `-73.792729936901 Ha`, gap to HF `+0.211466794700 Ha`, change vs previous lr2e-4 last1000 `-0.540753149490 Ha`, block stderr `0.067802 Ha`, pmove `0.5313`.
- first_half: mean `-73.627377030407 Ha`, gap to HF `+0.376819701194 Ha`, change vs previous lr2e-4 last1000 `-0.375400242996 Ha`, block stderr `0.055967 Ha`, pmove `0.5306`.
- second_half: mean `-73.958082843396 Ha`, gap to HF `+0.046113888205 Ha`, change vs previous lr2e-4 last1000 `-0.706106055985 Ha`, block stderr `0.021705 Ha`, pmove `0.5320`.
- last_5000: mean `-74.014606121362 Ha`, gap to HF `-0.010409389761 Ha`, change vs previous lr2e-4 last1000 `-0.762629333951 Ha`, block stderr `0.016688 Ha`, pmove `0.5329`.
- last_2000: mean `-74.028304211607 Ha`, gap to HF `-0.024107480006 Ha`, change vs previous lr2e-4 last1000 `-0.776327424196 Ha`, block stderr `0.021423 Ha`, pmove `0.5340`.
- last_1000: mean `-74.032080094067 Ha`, gap to HF `-0.027883362466 Ha`, change vs previous lr2e-4 last1000 `-0.780103306656 Ha`, block stderr `0.033767 Ha`, pmove `0.5328`.
- last_500: mean `-74.020123304181 Ha`, gap to HF `-0.015926572580 Ha`, change vs previous lr2e-4 last1000 `-0.768146516770 Ha`, block stderr `0.007402 Ha`, pmove `0.5337`.
- last_100: mean `-73.922076287965 Ha`, gap to HF `+0.082120443636 Ha`, change vs previous lr2e-4 last1000 `-0.670099500554 Ha`, block stderr `0.089908 Ha`, pmove `0.5408`.

## Block Means

- Block 01 steps 10000-10999: energy `-73.281533811309 Ha`, gap `+0.722662920292 Ha`, variance `42.2553`, pmove `0.5306`.
- Block 02 steps 11000-11999: energy `-73.492824917565 Ha`, gap `+0.511371814036 Ha`, variance `39.4586`, pmove `0.5280`.
- Block 03 steps 12000-12999: energy `-73.513922632623 Ha`, gap `+0.490274098978 Ha`, variance `39.6394`, pmove `0.5278`.
- Block 04 steps 13000-13999: energy `-73.586065636139 Ha`, gap `+0.418131095462 Ha`, variance `42.3175`, pmove `0.5320`.
- Block 05 steps 14000-14999: energy `-73.564766352480 Ha`, gap `+0.439430379121 Ha`, variance `45.0270`, pmove `0.5284`.
- Block 06 steps 15000-15999: energy `-73.671755644365 Ha`, gap `+0.332441087236 Ha`, variance `33.1515`, pmove `0.5330`.
- Block 07 steps 16000-16999: energy `-73.711056847229 Ha`, gap `+0.293139884372 Ha`, variance `34.6350`, pmove `0.5278`.
- Block 08 steps 17000-17999: energy `-73.797046693813 Ha`, gap `+0.207150037788 Ha`, variance `38.3097`, pmove `0.5344`.
- Block 09 steps 18000-18999: energy `-73.764462603955 Ha`, gap `+0.239734127646 Ha`, variance `32.8128`, pmove `0.5307`.
- Block 10 steps 19000-19999: energy `-73.890335164587 Ha`, gap `+0.113861567014 Ha`, variance `32.6863`, pmove `0.5329`.
- Block 11 steps 20000-20999: energy `-73.865700291719 Ha`, gap `+0.138496439882 Ha`, variance `30.5271`, pmove `0.5293`.
- Block 12 steps 21000-21999: energy `-73.841230955527 Ha`, gap `+0.162965776074 Ha`, variance `35.3120`, pmove `0.5306`.
- Block 13 steps 22000-22999: energy `-73.920675471812 Ha`, gap `+0.083521259789 Ha`, variance `29.9338`, pmove `0.5303`.
- Block 14 steps 23000-23999: energy `-73.941099226598 Ha`, gap `+0.063097505002 Ha`, variance `28.4765`, pmove `0.5334`.
- Block 15 steps 24000-24999: energy `-73.939091881491 Ha`, gap `+0.065104850110 Ha`, variance `28.6096`, pmove `0.5318`.
- Block 16 steps 25000-25999: energy `-73.975480746642 Ha`, gap `+0.028715984959 Ha`, variance `28.4878`, pmove `0.5315`.
- Block 17 steps 26000-26999: energy `-74.010856666656 Ha`, gap `-0.006659935055 Ha`, variance `26.4255`, pmove `0.5337`.
- Block 18 steps 27000-27999: energy `-74.030084770297 Ha`, gap `-0.025888038696 Ha`, variance `28.9962`, pmove `0.5314`.
- Block 19 steps 28000-28999: energy `-74.024528329147 Ha`, gap `-0.020331597546 Ha`, variance `29.0116`, pmove `0.5351`.
- Block 20 steps 29000-29999: energy `-74.032080094067 Ha`, gap `-0.027883362466 Ha`, variance `26.4263`, pmove `0.5328`.

## Interpretation

The additional 20000 optimization steps substantially improved the wavefunction. The second-half mean is within about 0.046 Ha of the same-cell HF reference, and the last 5000 steps are slightly below the HF reference. Because this is still a training trajectory with Monte Carlo noise and a variational estimate should be checked independently, the next required action is a fixed-checkpoint evaluation of `qmcjax_ckpt_029999.npz`.
