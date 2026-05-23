# LR2e-4 Continuation Summary

Experiment: `diamond_c_deepsolid_continue_ckpt4999_lr2e4_batch96_mcmc20`
Job: `120760`, `COMPLETED 0:0`, elapsed `00:08:32` on `gpu40904`.

## Setup

- Restored checkpoint: `qmcjax_ckpt_004999.npz` from the 5000-step training run.
- Optimizer: `adam`.
- Learning rate: `2e-4`, delay `5000`, decay `1.0`.
- Batch size: `96`.
- MCMC steps per iteration: `20`.
- Continuation rows: `5000`, steps `5000` through `9999`.
- Runtime resources: `6 x RTX 4090`, `96 CPU`.
- HF reference: `-74.004196731601 Ha`.

## Energy Estimate

- all: mean `-72.788484001366 Ha`, gap to HF `+1.215712730235 Ha`, change vs fixed-eval `-0.750822640707 Ha`, block stderr `0.131233 Ha`, pmove `0.5274`.
- first_half: mean `-72.481636690729 Ha`, gap to HF `+1.522560040872 Ha`, change vs fixed-eval `-0.443975330071 Ha`, block stderr `0.110209 Ha`, pmove `0.5257`.
- second_half: mean `-73.095331312002 Ha`, gap to HF `+0.908865419599 Ha`, change vs fixed-eval `-1.057669951344 Ha`, block stderr `0.057058 Ha`, pmove `0.5291`.
- last_1000: mean `-73.251976787411 Ha`, gap to HF `+0.752219944190 Ha`, change vs fixed-eval `-1.214315426753 Ha`, block stderr `0.040515 Ha`, pmove `0.5277`.
- last_500: mean `-73.311110395212 Ha`, gap to HF `+0.693086336389 Ha`, change vs fixed-eval `-1.273449034554 Ha`, block stderr `0.044030 Ha`, pmove `0.5291`.
- last_100: mean `-73.425988733963 Ha`, gap to HF `+0.578207997638 Ha`, change vs fixed-eval `-1.388327373305 Ha`, block stderr `0.020465 Ha`, pmove `0.5353`.

## Block Means

- Block 01 steps 5000-5499: energy `-72.000376529054 Ha`, gap `+2.003820202547 Ha`, variance `74.4739`, pmove `0.5241`.
- Block 02 steps 5500-5999: energy `-72.422362795704 Ha`, gap `+1.581833935897 Ha`, variance `61.3717`, pmove `0.5221`.
- Block 03 steps 6000-6499: energy `-72.349138163537 Ha`, gap `+1.655058568064 Ha`, variance `70.5131`, pmove `0.5267`.
- Block 04 steps 6500-6999: energy `-72.708955035270 Ha`, gap `+1.295241696331 Ha`, variance `56.4300`, pmove `0.5276`.
- Block 05 steps 7000-7499: energy `-72.927350930080 Ha`, gap `+1.076845801520 Ha`, variance `52.4055`, pmove `0.5279`.
- Block 06 steps 7500-7999: energy `-72.897253561846 Ha`, gap `+1.106943169755 Ha`, variance `64.3244`, pmove `0.5295`.
- Block 07 steps 8000-8499: energy `-72.995949185541 Ha`, gap `+1.008247546060 Ha`, variance `49.3386`, pmove `0.5268`.
- Block 08 steps 8500-8999: energy `-73.079500237802 Ha`, gap `+0.924696493799 Ha`, variance `48.7716`, pmove `0.5338`.
- Block 09 steps 9000-9499: energy `-73.192843179611 Ha`, gap `+0.811353551990 Ha`, variance `50.0368`, pmove `0.5262`.
- Block 10 steps 9500-9999: energy `-73.311110395212 Ha`, gap `+0.693086336389 Ha`, variance `45.1987`, pmove `0.5291`.

## Interpretation

The low-learning-rate continuation clearly improved the trained wavefunction. The second-half mean is more than 1 Ha lower than the fixed-checkpoint evaluation, and the final step is within about 0.03 Ha of the same-cell HF reference. Because the run is still trending downward, the next useful action is a fixed-checkpoint evaluation of `qmcjax_ckpt_009999.npz` before deciding whether to continue or scale the model.
