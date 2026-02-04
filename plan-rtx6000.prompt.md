## Plan: Generate Orion + MPS results on RTX 6000

Use the existing experiment harnesses, but point them at kernel/profile files generated on your Quadro RTX 6000 (sm75). You can temporarily reuse V100 kernel files just to validate the pipeline runs, but for publishable/meaningful results you should re-profile because Orion’s scheduling decisions depend directly on those per-kernel measurements.

### Steps
1. Identify the experiment entrypoints and configs you’ll use (recommended: [h100_results/inf_inf_updated/run_orion.py](h100_results/inf_inf_updated/run_orion.py) + [h100_results/inf_inf_updated/run_mps.py](h100_results/inf_inf_updated/run_mps.py), configured via [h100_results/inf_inf_updated/config_files/](h100_results/inf_inf_updated/config_files/)).
2. Build the Orion interception/scheduler libraries and verify `LD_PRELOAD` works (see build script [compile.sh](compile.sh) and run notes in [README.md](README.md)).
3. Re-profile on RTX 6000 to generate new kernel info files (drive scripts in [profiling/](profiling/) and produce a new folder under [benchmarking/model_kernels/](benchmarking/model_kernels/) such as `rtx6000/`).
4. Update experiment JSON configs to point `kernel_file` at your RTX 6000 kernel info paths (configs live in [h100_results/inf_inf_updated/config_files/](h100_results/inf_inf_updated/config_files/); the runner ultimately calls [benchmarking/launch_jobs.py](benchmarking/launch_jobs.py)).
5. Run Orion experiments and collect outputs (use the wrappers in [h100_results/inf_inf_updated/](h100_results/inf_inf_updated/) which create `results/` and copy client outputs).
6. Start CUDA MPS, then run the baseline MPS harness and collect outputs (MPS control script [related/baselines/run_wrapper.sh](related/baselines/run_wrapper.sh), baseline main [related/baselines/main.py](related/baselines/main.py), config [related/baselines/config.yaml](related/baselines/config.yaml)).

### Further Considerations
1. Quick sanity run: reuse V100 kernel files first, then re-profile for final numbers.
2. RTX 6000 specifics: set profiling occupancy params for sm75 (not V100 defaults) in the scripts under [profiling/postprocessing/](profiling/postprocessing/).
3. Confirm which models/batch sizes are in scope by inspecting the configs you plan to run in [h100_results/inf_inf_updated/config_files/](h100_results/inf_inf_updated/config_files/).
