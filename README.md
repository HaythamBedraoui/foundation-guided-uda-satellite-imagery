# Foundation-Guided Unsupervised Domain Adaptation for Vehicle Detection in Middle Eastern Satellite Imagery

This repository contains the code and paper for a study extending the [VME dataset benchmark](https://www.nature.com/articles/s41597-025-04567-y) (Al-Emadi et al., Scientific Data 2025) and the [RWDS benchmark](https://openaccess.thecvf.com/content/CVPR2025/papers/Al-Emadi_Benchmarking_Object_Detectors_under_Real-World_Distribution_Shifts_in_Satellite_Imagery_CVPR_2025_paper.pdf) (Al-Emadi et al., CVPR 2025).

The core question: can a car detector trained on xView (Western satellite imagery) be adapted to Middle Eastern imagery without using any target-domain labels?

---

## What this project does

**Phase 1** reproduces the VME source-only baseline: a TOOD detector trained on xView tiles reaches 38.2 mAP50 on VME, matching the published number exactly. The RWDS robustness finding is also reproduced — Grounding DINO drops only 7.8% going from xView to VME, versus 34.9% for the trained detector.

**Phase 2** evaluates Grounding DINO zero-shot on VME across five text prompts. The plain prompt "car" is the strongest at 14.7 mAP50; adding domain context ("small car in aerial image") hurts performance. The best-prompt detections are saved as pseudo-label candidates for Phase 3.

**Phase 3** runs foundation-guided self-training. The source detector pseudo-labels the unlabeled VME training images (threshold selected on VME val by F1 maximisation), Grounding DINO detections are fused in to improve recall, and a student is fine-tuned with weak/strong augmentation asymmetry on a source-anchored mixture. An ablation isolates the effect of foundation guidance by training an identical student on source-only pseudo-labels.

**Phase 4** produces the paper's figures and tables from the saved results.

---

## Main results

| Method | VME mAP50 | VME mAP |
|---|---|---|
| Grounding DINO (zero-shot) | 14.7 | 4.5 |
| TOOD source-only (xView→VME) | 38.2 | 13.4 |
| TOOD + self-training | 40.0 | 13.8 |
| TOOD + foundation-guided (ours) | 39.9 | 13.8 |
| TOOD oracle (standard) | 55.6 | 23.2 |

Self-training improves VME recall from 0.26 to 0.48 without target labels. Foundation guidance does not help at this operating point — the source detector at threshold 0.15 already produces 19.1 pseudo-boxes per image, leaving the foundation model (≈1 box per image) too little room to contribute. The ablation confirms this: fused and source-only pseudo-labels produce nearly identical results (39.9 vs 40.0 mAP50). Adaptation also causes significant source-domain forgetting (xView drops from 58.8 to 18.5 mAP50), which we document as a limitation.

---

## Data

**xView** — download from [xviewdataset.org](http://xviewdataset.org) (free registration required). Place images at `QCRI-CV/xview/train_images/` and the GeoJSON labels at `QCRI-CV/xview/train_labels/xView_train.geojson`.

**VME** — download from [Zenodo record 14185684](https://zenodo.org/records/14185684). Place the contents at `QCRI-CV/VME_CDSI_datasets/`.

Both datasets provide their own construction scripts for generating the car-only COCO splits used here. The notebooks use the official scripts from the VME repository without modification.

---

## Setup

Tested on Google Colab with an A100 GPU and PyTorch 2.4.0. The prebuilt mmcv 2.1.0 CUDA 12.1 wheels are incompatible with sm_120 (H100/Blackwell) — use an A100 runtime for Phases 1 and 3.

Section 1 of each notebook handles the environment. On the first run, use cell 1.4a (fresh install, ~50 min for mmcv compilation) then cell 1.8 to save the package cache to Drive. On subsequent sessions, cell 1.4b restores from the cache in a few minutes.

```
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install mmengine
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
pip install mmdet==3.3.0 sahi transformers pycocotools wandb
```

---

## Running the notebooks

Run in order. Each notebook is self-contained and writes its outputs to `QCRI-CV/phaseN/`.

| Notebook | What it does | GPU needed |
|---|---|---|
| `01_tood_baseline_xview_to_vme.ipynb` | Train TOOD on xView, evaluate on VME and xView | Yes (A100, ~9h) |
| `02_gdino_zeroshot_vme.ipynb` | Grounding DINO zero-shot + prompt ablation | Yes |
| `03_foundation_guided_selftraining_vme.ipynb` | Pseudo-labels + self-training + ablation | Yes (A100, ~10h total) |
| `04_analysis_and_results.ipynb` | Figures and tables | No |

Section 4.3 in notebooks 1 and 3 resumes automatically if the runtime disconnects — just re-run the cell.

---

## Repository structure

```
├── 01_tood_baseline_xview_to_vme.ipynb
├── 02_gdino_zeroshot_vme.ipynb
├── 03_foundation_guided_selftraining_vme.ipynb
├── 04_analysis_and_results.ipynb
├── paper.pdf
└── requirements.txt
```

---

## Citation

If you use this code or build on these results, please also cite the two papers this work is based on:

```bibtex
@article{alemadi2025vme,
  title={VME: A Satellite Imagery Dataset and Benchmark for Detecting Vehicles in the Middle East and Beyond},
  author={Al-Emadi, Noora and Weber, Ingmar and Yang, Yin and Ofli, Ferda},
  journal={Scientific Data},
  volume={12},
  number={1},
  pages={500},
  year={2025}
}

@inproceedings{alemadi2025rwds,
  title={Benchmarking Object Detectors under Real-World Distribution Shifts in Satellite Imagery},
  author={Al-Emadi, Sara A. and Yang, Yin and Ofli, Ferda},
  booktitle={CVPR},
  year={2025}
}
```

---

## Author

Haytham Bedraoui — ENSICAEN, Caen, France
haytham.bedraoui@ecole.ensicaen.fr
