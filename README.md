<h1 align="center">Towards Fast and Scalable Normal Integration using Continuous Components</h1>
<p align="center">
<strong><a href="https://scholar.google.com/citations?user=qwSANZoAAAAJ&hl=en">Francesco Milano</a></strong>, <strong><a href="https://jenjenchung.github.io/anthropomorphic/">Jen Jen Chung</a></strong>, <strong><a href="http://www.ott.ai/">Lionel Ott</a></strong>, <strong><a href="https://asl.ethz.ch/">Roland Siegwart</a></strong>
</p>

<h2 align="center">WACV 2026</h2>
<h3 align="center"><a href="https://arxiv.org/abs/2510.11508">Paper</a> | <a href="https://www.youtube.com/watch?v=dZQAswCNM2E">Video</a></h3>


<p align="center">
  <a href="">
    <img src="./assets/method_overview.png" alt="Towards Fast and Scalable Normal Integration using Continuous Components" width="100%">
  </a>
</p>

Surface normal integration is a fundamental problem in computer vision, dealing with the objective of reconstructing a surface from its corresponding normal map. Existing approaches require an iterative global optimization to jointly estimate the depth of each pixel, which scales poorly to larger normal maps. In this paper, we address this problem by recasting normal integration as the estimation of relative scales of continuous components. By constraining pixels belonging to the same component to jointly vary their scale, we drastically reduce the number of optimization variables. Our framework includes a heuristic to accurately estimate continuous components from the start, a strategy to rebalance optimization terms, and a technique to iteratively merge components to further reduce the size of the problem. Our method achieves state-of-the-art results on the standard normal integration benchmark in as little as a few seconds and achieves one-order-of-magnitude speedup over pixel-level approaches on large-resolution normal maps.

## Installation
The reference code in this repository was tested on Ubuntu 20.04, using a Python 3.11.13 virtual environment. The external libraries can be installed into the virtual environments as follows, where it is assumed that the virtual environment has been sourced and where we refer to the root folder of this repo as `${REPO_ROOT}`:
1. Install the packages specified in the [`requirements.txt`](./requirements.txt) file:
    ```bash
    cd ${REPO_ROOT};
    pip install -r requirements.txt
    ```
2. Install PyTorch. The code was tested with PyTorch 2.2.0 with CUDA 11.8; if you are using a different configuration try installing a different version following the [official instructions](https://pytorch.org/get-started/locally/):
    ```bash
    pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118
    ```
3. Install PyTorch Scatter, matching the PyTorch version you installed above, see [here](https://github.com/rusty1s/pytorch_scatter?tab=readme-ov-file#installation):
    ```bash
    pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0%2Bcu118.html
    ```
4. Clone and install the submodule [`surface_normal_integration`](./third_party/):
    ```bash
    cd ${REPO_ROOT};
    git submodule update --init --recursive;
    pip install -e third_party/surface_normal_integration;
    ```
5. Install the current repo as a package:
    ```bash
    cd ${REPO_ROOT};
    pip install -e .;
    ```

## Data setup
### DiLiGenT dataset
Refer to our previous project [`surface_normal_integration`](https://github.com/facebookresearch/surface_normal_integration?tab=readme-ov-file#diligent-download) for instructions on how to download and set up the DiLiGenT dataset.

### Custom data
Custom data needs to be provided in the same format as the DiLiGenT dataset. For more details, refer to the `CustomObject` class in [`datasets/custom.py`](./normal_integration_continuous_components/datasets/custom.py) and to the examples in the [`example_data`](./example_data/) folder, which include normal maps used in the paper.

## Basic usage
Results similar to the experiments in the paper can be obtained with the following command:
```bash
cd ${REPO_ROOT};
python normal_integration_continuous_components/main.py \
    --data_dir ${DATASET_ROOT} \
    --obj_type ${OBJ_TYPE} \
    --obj_name ${OBJ_NAME} \
    --output_subfolder ${OUTPUT_SUBFOLDER} \
    --threshold_continuity_criterion_deg ${THRESHOLD_CONTINUITY_CRITERION_DEG} \
    --outlier_reweighting_type ${OUTLIER_REWEIGHTING_TYPE} \
    --threshold_surely_continuous_residual ${THRESHOLD_SURELY_CONTINUOUS_RESIDUAL} \
    --threshold_noncontinuous_residual ${THRESHOLD_NONCONTINUOUS_RESIDUAL} \
    --component_vis_log_freq ${COMPONENT_VIS_LOG_FREQ}
```
where
- `${DATASET_ROOT}` is the root folder of the dataset (cf. [here](https://github.com/facebookresearch/surface_normal_integration?tab=readme-ov-file#diligent-download) for an example with the DiLiGenT dataset);
- `${OBJ_TYPE}` is `diligent` for the experiments on DiLiGenT and `custom` for the experiments on custom data;
- `${OBJ_NAME}` is the object name (_e.g.,_ `bear` in the DiLiGenT dataset, or `bedroom` for the custom data);
- `${THRESHOLD_CONTINUITY_CRITERION_DEG}` is the max angle in degrees that the normals at two neighboring pixels can have for the edge between the two pixels to be considered continuous; cf. $\theta_c$ in the paper. If `"None"`, no initial decomposition based on normal similarity is computed (_i.e._, per-pixel decomposition is used);
- `${OUTLIER_REWEIGHTING_TYPE}` defines the type of outlier reweighting (cf. Sec. 3.4 and Appendix C in the paper). If `'none'`, no outlier reweighting is applied. If `'hard'`, after the initial components have been made as globally continuous as possible, all residuals larger in magnitude than `${THRESHOLD_NONCONTINUOUS_RESIDUAL}` are assigned hard weight `0` during later stages of the optimization. If `'soft'`, a soft sigmoid-based weight is multiplied to the original residual weight, so that `-log10(abs(residual))` is linearly mapped to the argument of the sigmoid, with the mapping such that `-log(${THRESHOLD_NONCONTINUOUS_RESIDUAL})` is mapped to `-4` (which maps to `~0.02` through the sigmoid function) and `-log(${THRESHOLD_SURELY_CONTINUOUS_RESIDUAL})` to `4` (which maps to `~0.98` through the sigmoid function).
- `${THRESHOLD_SURELY_CONTINUOUS_RESIDUAL}`: Cf. `${OUTLIER_REWEIGHTING_TYPE}` and $U$ in the paper;
- `${THRESHOLD_NONCONTINUOUS_RESIDUAL}`: Cf. `${OUTLIER_REWEIGHTING_TYPE}` and $L$ in the paper;
- `${COMPONENT_VIS_LOG_FREQ}`: Frequency (in number of meta-optimization iterations) with which a connected-component image is saved to file. If `'only_after_merge'`, images are saved only after a merge operation is performed. If `'None'`, no saving is performed;
- `${OUTPUT_SUBFOLDER}` is the path to the folder that should store the output of the experiments. For each experiment, a subfolder is created that is indexed by the experiment's starting time and by the object name.

Additional optional flags include:
- `--use_log_depth_formulation`: If passed, the log-depth formulation of previous methods is used instead of our formulation based on relative log scales;
- `--tol_relative_energy_change ${TOL_RELATIVE_ENERGY_CHANGE}`: Threshold for the relative change in the optimization energy in later stages of the optimization; if the optimization energy between two subsequent iterations changes in relative terms by less than this threshold, meta-optimization is stopped early and a merging is performed; cf. $\Delta E_\mathrm{max}$ in the paper (default: `1.0e-3`);
- `--freq_merging ${FREQ_MERGING}`: Minimum frequency of component merging in number of iterations; cf. $\mathrm{freq}_\mathrm{merging}$ in the paper (default: `5`). If no merging is performed (as is the case when `--use_log_depth_formulation` is passed), it acts as the maximum number of iterations (cf. $T$ in the paper);
- `--allow_only_two_sided_intercomponent_edges`: If passed, in forming the continuous components a pair $(a, b)$ of pixels is only considered if the pixel $-b$ is also in the valid-pixel mask;
- `--use_only_horizontal_vertical`: If passed, in forming the continuous components $4$-pixel connectivity is used (_note_: $8$-pixel connectivity is still used in the computation of the residuals for the optimization);
- `--initial_filling_type ${INITIAL_FILLING_TYPE}`: Type of per-component filling. If `"joint_optimize"`, a single, global optimization problem (but still without intercomponent edges) is run; if `"parallel_optimize"`, separate, per-component optimization problems are run in parallel (default: `"parallel_optimize"`).

- `--compute_min_theoretical_made`: If passed, the minimum theoretical MADE is computed after filling the components (cf. Appendix E in the paper);
- `--log_intermediate_mades`: If passed, the MADE values at intermediate steps of the optimization are computed and logged (_note_: this will slow down execution);
- `--log_timings`: If passed, per-step execution times are logged in a `timings_log.txt` file (cf. Appendix B in the paper).

- `--min_res_th ${MIN_RES_TH}`: For all instances where a connected-component computation is required, all input values below this threshold in magnitude are remapped to have magnitude equal to this value. This is necessary because the `scipy.sparse` algorithms interpret very small values as the absence of an edge (default: `5.0e-8`);
- `--cg_max_iter ${CG_MAX_ITER}`: Maximum number of conjugate-gradient iterations (default: `5000`);
- `--cg_tol ${CG_TOL}`: Relative convergence tolerance for conjugate-gradient optimization (default: `1.0e-3`).

Example run on DiLiGenT:
```bash
cd ${REPO_ROOT};
python normal_integration_continuous_components/main.py \
    --data_dir ${DATASET_ROOT} \
    --obj_type diligent \
    --obj_name bear \
    --output_subfolder ${OUTPUT_SUBFOLDER} \
    --threshold_continuity_criterion_deg 3.5 \
    --outlier_reweighting_type soft \
    --threshold_surely_continuous_residual 1.0e-5 \
    --threshold_noncontinuous_residual 1.0e-3 \
    --component_vis_log_freq 1 \
    --log_timings
```

Example run on custom data (similar to Fig. 4 in the paper):
```bash
cd ${REPO_ROOT};
# BlenderProc renderings (obj_name: `bedroom` or `living_room`).
python normal_integration_continuous_components/main.py \
  --data_dir ${REPO_ROOT}/example_data/BlenderProc \
  --obj_type custom \
  --obj_name bedroom \
  --output_subfolder ${OUTPUT_SUBFOLDER} \
  --threshold_continuity_criterion_deg 2.0 \
  --freq_merging 5 \
  --threshold_noncontinuous_residual 1.0e-3 \
  --threshold_surely_continuous_residual 1.0e-5 \
  --component_vis_log_freq 1 \
  --log_timings
# DSINE predictions (obj_name: `seafloor` or `wedding_cake`).
python normal_integration_continuous_components/main.py \
  --data_dir ${REPO_ROOT}/example_data/DSINE \
  --obj_type custom \
  --obj_name seafloor \
  --output_subfolder ${OUTPUT_SUBFOLDER} \
  --threshold_continuity_criterion_deg 2.0 \
  --freq_merging 5 \
  --threshold_noncontinuous_residual 1.0e-3 \
  --threshold_surely_continuous_residual 1.0e-5 \
  --component_vis_log_freq 1 \
  --log_timings
```


## Citation
If you find our code or paper useful, please cite:

```bibtex
@inproceedings{Milano2026TowardsFastScalableNormalIntegration,
  author    = {Milano, Francesco and Chung, Jen Jen and Ott, Lionel and Siegwart, Roland},
  title     = {{Towards Fast and Scalable Normal Integration using Continuous Components}},
  booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2026}
}
```

## License
The code in this project is GPLv3 licensed, as found in the LICENSE file.


## Acknowledgements
This repository depends on the code base from our previous project <a href="https://github.com/facebookresearch/surface_normal_integration">surface_normal_integration</a>, forked to implement a slight refactoring and minor modifications (cf. folder [`third_party/surface_normal_integration/`](./third_party/surface_normal_integration/)). Parts of the optimization procedure in the [main script](./normal_integration_continuous_components/main.py) are based on <a href="https://github.com/xucao-42/bilateral_normal_integration/">BiNI</a>.
