# Fixed-Ground Explicit-Only Policy

Date: 2026-06-01, Asia/Shanghai

The default excited-state routes are the DeepQMC-aligned PsiFormer/FermiNet
native `vmc_overlap` methods, including attention QKV comparisons such as
`fused_qkv` versus upstream-shaped `ferminet` Q/K/V handling.

Fixed-ground is not a default excited-state method selection. Do not select,
configure, or submit fixed-ground jobs unless the user explicitly asks for
`fixed-ground`.

This policy was added after clarifying task 0106: "the two methods" means the
two attention QKV handling routes, not fixed-ground.
