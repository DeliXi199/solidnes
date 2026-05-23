# Fixed-Checkpoint Evaluation Summary

Experiment: `diamond_c_deepsolid_evaluate_ckpt4999_batch96_mcmc20`
Job: `120750`, `COMPLETED 0:0`, elapsed `00:06:09` on `gpu004`.

## Setup

- Restored checkpoint: `qmcjax_ckpt_004999.npz` from the 5000-step training run.
- Optimizer: `none`.
- Batch size: `96`.
- MCMC steps per iteration: `20`.
- Evaluation rows: `5000`, steps `5000` through `9999`.
- HF reference: `-74.004196731601 Ha`.

## Energy Estimate

- all: mean `-72.037661360658 Ha`, gap to HF `+1.966535370943 Ha`, naive stderr `0.011909 Ha`, block stderr `0.021710 Ha`, pmove `0.5164`.
- second_half: mean `-71.995671542752 Ha`, gap to HF `+2.008525188849 Ha`, naive stderr `0.016613 Ha`, block stderr `0.033594 Ha`, pmove `0.5154`.
- last_1000: mean `-72.012440707173 Ha`, gap to HF `+1.991756024428 Ha`, naive stderr `0.026603 Ha`, block stderr `0.058677 Ha`, pmove `0.5140`.
- last_500: mean `-72.075355417964 Ha`, gap to HF `+1.928841313637 Ha`, naive stderr `0.037902 Ha`, block stderr `0.065258 Ha`, pmove `0.5148`.

## Block Means

- Block 01 steps 5000-5499: energy `-72.042163367450 Ha`, gap `+1.962033364151 Ha`, variance `77.1148`, pmove `0.5156`.
- Block 02 steps 5500-5999: energy `-72.093155943119 Ha`, gap `+1.911040788482 Ha`, variance `69.2032`, pmove `0.5184`.
- Block 03 steps 6000-6499: energy `-72.125467814207 Ha`, gap `+1.878728917394 Ha`, variance `70.1588`, pmove `0.5195`.
- Block 04 steps 6500-6999: energy `-72.082490408945 Ha`, gap `+1.921706322656 Ha`, variance `81.5912`, pmove `0.5181`.
- Block 05 steps 7000-7499: energy `-72.054978359102 Ha`, gap `+1.949218372498 Ha`, variance `87.9023`, pmove `0.5160`.
- Block 06 steps 7500-7999: energy `-72.070710640768 Ha`, gap `+1.933486090833 Ha`, variance `77.4823`, pmove `0.5168`.
- Block 07 steps 8000-8499: energy `-71.956844680059 Ha`, gap `+2.047352051542 Ha`, variance `75.8355`, pmove `0.5174`.
- Block 08 steps 8500-8999: energy `-71.925920978587 Ha`, gap `+2.078275753014 Ha`, variance `75.6408`, pmove `0.5147`.
- Block 09 steps 9000-9499: energy `-71.949525996382 Ha`, gap `+2.054670735219 Ha`, variance `74.6734`, pmove `0.5131`.
- Block 10 steps 9500-9999: energy `-72.075355417964 Ha`, gap `+1.928841313637 Ha`, variance `83.8722`, pmove `0.5148`.

## Interpretation

The fixed wavefunction estimate is finite and much closer to the same-cell HF reference than earlier short runs, but it remains roughly 2 Ha above HF. The single minimum is not the estimator; the block mean is the meaningful quantity for this fixed-parameter evaluation.
