# UAE Licence Plate Detection and Recognition

A computer-vision pipeline for UAE licence plate detection, plate-text recognition, and emirate classification.

The project compares multiple detector architectures on the same frozen train/validation/test split and extends the detector with OCR and seven-emirate classification.

[Project overview and examples](https://github.com/azimaus970/UAE-Plate-Recognition/blob/main/README.md)

## Team

This project was developed by:

- **Adrish Danka**
- **Azim Mohamed**
- **Mohammed Rashdan**
- **Sam Tom**

## Project Overview

The pipeline uses RF-DETR Medium to detect a plate and expands the bounding box before cropping. OCR v2 and the seven-class ResNet18 classifier read the text and predict the emirate from that crop. The parser then splits the recognized string into the plate code and registration number using training-derived rules. The script saves an annotated image, plate crops, and JSON.

The detector comparison covers RF-DETR Medium, RT-DETR-L, and YOLOv8n. The RF-DETR study also includes a run without augmentation.

## Dataset

The frozen detector dataset contains:

| Split | Images | Licence-plate boxes |
|---|---:|---:|
| Train | 6,738 | 9,415 |
| Validation | 1,440 | 1,525 |
| Test | 1,432 | 1,511 |
| **Total** | **9,610** | **12,451** |

The detector task uses one class:

```text
license_plate
```

Cross-split duplicate checks found:

- Exact duplicates: 0
- Flagged perceptual duplicate candidates: 0

The full image dataset is distributed separately. The root of this repository includes the YOLO labels and COCO annotations. [Dataset guide](https://github.com/azimaus970/UAE-Plate-Recognition/blob/main/PREPROCESSING.md) · [Dataset attribution](https://github.com/azimaus970/UAE-Plate-Recognition/blob/main/recognition/docs/DATASET_ATTRIBUTION.md)

## Main Results

### Detector comparison

All three models were evaluated on the same frozen 1,432-image test split.

| Model | mAP@50:95 | mAP@50 | Precision | Recall |
|---|---:|---:|---:|---:|
| RF-DETR Medium | 0.8664 | 0.9965 | 0.9914* | 0.9921* |
| RT-DETR-L | 0.8639 | 0.9944 | 0.9774 | 0.9921 |
| YOLOv8n | 0.8607 | 0.9949 | 0.9911 | 0.9914 |

*RF-DETR precision and recall were measured at a confidence threshold selected on the validation set and frozen before test evaluation.*

### RF-DETR augmentation ablation

| Configuration | mAP@50:95 | mAP@50 | mAP@75 | AR@100 |
|---|---:|---:|---:|---:|
| With augmentation | 0.8664 | 0.9965 | 0.9844 | 0.9027 |
| Without augmentation | 0.8666 | 0.9903 | 0.9777 | 0.9060 |

The overall mAP@50:95 difference was negligible. The augmented model performed better at IoU 0.50 and 0.75, while the non-augmented model produced slightly higher average recall.

### OCR crop evaluation

| Model and test subset | Crops | Exact plate match | Character accuracy | Plate-length accuracy |
|---|---:|---:|---:|---:|
| OCR v1, original | 1,017 | 97.94% | 99.72% | 99.71% |
| OCR v2, original | 1,017 | 98.23% | 99.77% | 99.90% |
| OCR v2, numeric-prefix Abu Dhabi | 173 | 86.13% | 97.56% | 95.38% |

OCR v2 retained the original test subset and added a separate evaluation for numeric-prefix Abu Dhabi plates. Exact matches on that subset remain harder. [OCR evaluation notes](https://github.com/azimaus970/UAE-Plate-Recognition/blob/main/recognition/docs/OCR_END_TO_END_REPORT_ADDENDUM.md) · [Abu Dhabi error examples](https://github.com/azimaus970/UAE-Plate-Recognition/blob/main/recognition/results/figures/ocr/ocr_uae_v2_abu_dhabi_test-incorrect_examples.jpg)

### Emirate classifier

| Metric | Result |
|---|---:|
| Test crops | 1,466 |
| Accuracy | 99.59% |
| Balanced accuracy | 97.71% |
| Macro F1 | 98.54% |
| Weighted F1 | 99.59% |

The classifier predicts:

- Dubai
- Abu Dhabi
- Sharjah
- Ajman
- Fujairah
- Ras Al Khaimah
- Umm Al Quwain

Because the dataset is imbalanced, balanced accuracy and macro F1 are more informative than accuracy alone.

## Repository Structure

```text
recognition/
├── README.md
├── START_HERE.md
├── requirements.txt
├── .gitignore
├── configs/
├── docs/
├── scripts/
└── results/
    ├── metrics/
    ├── figures/
    └── examples/
```

### Important scripts

```text
scripts/
├── preprocess_dataset.py
├── preprocessing_utils.py
├── validate_dataset.py
├── check_split_leakage.py
├── prepare_rfdetr_dataset.py
├── train_rfdetr_main.py
├── train_rfdetr_no_aug.py
├── evaluate_rfdetr_test.py
├── evaluate_rfdetr_no_aug.py
├── evaluate_fixed_threshold.py
├── benchmark_rfdetr.py
├── export_rfdetr_curves.py
├── train_rtdetr_l.py
├── build_ocr_dataset.py
├── build_ocr_v2_dataset.py
├── evaluate_ocr_test.py
├── build_emirate_dataset.py
├── finalize_emirate_labels.py
├── train_emirate_classifier.py
├── build_emirate_code_rules.py
└── run_final_uae_plate_pipeline_v2.py
```

## Environment Setup

The recorded recognition environment used Windows, Python 3.13.14, and PyTorch 2.11.0 with CUDA 12.8. Use the requirements file in `recognition/` for these commands. [Environment record](https://github.com/azimaus970/UAE-Plate-Recognition/blob/main/recognition/ENVIRONMENT.txt) · [PyTorch installation reference](https://pytorch.org/get-started/previous-versions/)

Start in the repository root in PowerShell.

```powershell
cd recognition
python -m venv rfdetr_env
.\rfdetr_env\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Run the commands below from `recognition/`. The repository-root environment is for dataset preprocessing. [Preprocessing setup](https://github.com/azimaus970/UAE-Plate-Recognition/blob/main/PREPROCESSING.md#quick-start)

## Expected Dataset Structure

The training scripts expect the following YOLO-format layout under `recognition/`.

```text
datasets/
└── uae_lp_v2_yolo/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

RF-DETR uses the corresponding COCO-format dataset produced by:

```powershell
python ".\scripts\prepare_rfdetr_dataset.py"
```

## Dataset Validation

```powershell
python ".\scripts\validate_dataset.py"
python ".\scripts\check_split_leakage.py"
```

## RF-DETR Training

### Main augmented model

```powershell
python ".\scripts\train_rfdetr_main.py"
```

### No-augmentation ablation

```powershell
python ".\scripts\train_rfdetr_no_aug.py"
```

### Test evaluation

```powershell
python ".\scripts\evaluate_rfdetr_test.py"
python ".\scripts\evaluate_fixed_threshold.py"
python ".\scripts\benchmark_rfdetr.py"
```

## RT-DETR-L Training

Validate the dataset and model setup:

```powershell
python ".\scripts\train_rtdetr_l.py" --check-only
```

Train and automatically evaluate the best checkpoint on the test split:

```powershell
python ".\scripts\train_rtdetr_l.py"
```

Main RT-DETR-L configuration:

- Input size: 576 × 576
- Maximum epochs: 20
- Physical batch size: 4
- Optimizer: AdamW
- Initial learning rate: 1e-4
- Early-stopping patience: 4
- Seed: 486

## OCR

### Build OCR v1 data

```powershell
python ".\scripts\build_ocr_dataset.py"
```

### Build OCR v2 data

```powershell
python ".\scripts\build_ocr_v2_dataset.py"
```

OCR v2 preserves the frozen split and adds strict numeric-prefix Abu Dhabi examples.

### Evaluate OCR

```powershell
python ".\scripts\evaluate_ocr_test.py" --help
```

See `docs/OCR_END_TO_END_REPORT_ADDENDUM.md` for the exact evaluation subsets and interpretation.

## Emirate Classification

```powershell
python ".\scripts\build_emirate_dataset.py"
python ".\scripts\finalize_emirate_labels.py"
python ".\scripts\train_emirate_classifier.py"
python ".\scripts\build_emirate_code_rules.py"
```

## Run the demo

The single-image demo uses RF-DETR Medium, OCR v2, ResNet18, and the parser rules. Install the [recognition environment](https://github.com/azimaus970/UAE-Plate-Recognition/blob/main/recognition/README.md#environment-setup) and place these files relative to `recognition/`.

| Required artifact | Placement |
|---|---|
| RF-DETR Medium weights | `runs/rfdetr_medium_main/checkpoint_best_total.pth` |
| OCR v2 weights | `runs/ocr_uae_v2_retry/<run>/best.keras` |
| OCR plate configuration | `runs/ocr_uae_v2_retry/<run>/plate_config.yaml`, next to `best.keras` |
| Emirate ResNet18 weights | `runs/emirate_resnet18_main/best.pt` |
| Emirate code rules | `results/emirate_code_rules.json` |

Model files are distributed separately. Check the folder against the list above before running. To regenerate the parser rules, run `python scripts/build_emirate_code_rules.py` with the recovered annotations CSV at `results/emirate_recovery/all_recovered_annotations.csv`. Bring a vehicle image as input. [Model download folder](https://drive.google.com/drive/folders/11NIEo7Cx430th-ADQH7vAt1NhTNl5QRZ) · [Checkpoint guide](https://github.com/azimaus970/UAE-Plate-Recognition/blob/main/recognition/docs/CHECKPOINTS.md)

```powershell
python scripts/run_final_uae_plate_pipeline_v2.py --image 'C:\path\to\vehicle.jpg'
```

The default command selects the newest `best.keras` under `runs/ocr_uae_v2_retry/`. For repeatable runs, select the exact checkpoint with `--ocr-model` and its YAML with `--plate-config`. Outputs go to `results/final_pipeline_demo_v2/<image-stem>/` and include `annotated.jpg`, `crops/`, and `result.json`. [Committed demo outputs](https://github.com/azimaus970/UAE-Plate-Recognition/tree/main/recognition/results/examples/final_pipeline)

To package a repeatable demo, keep the exact model and configuration files above together with a sample input and its expected output.

## Comparison checkpoints

The RF-DETR no-augmentation and RT-DETR-L weights are used for the detector comparisons. Their paths are listed in the [checkpoint guide](https://github.com/azimaus970/UAE-Plate-Recognition/blob/main/recognition/docs/CHECKPOINTS.md).

## Results and Report Material

Exact metrics:

```text
results/metrics/
```

Report-ready figures:

```text
results/figures/
```

Selected qualitative examples:

```text
results/examples/
```

Technical report sections and handoff notes:

```text
docs/
```

## Important Evaluation Notes

- Detector mAP, OCR accuracy, and emirate-classification F1 measure different tasks and should not be compared directly.
- The final-pipeline examples are qualitative. The reported accuracy scores are for individual stages.
- RF-DETR precision and recall use a validation-selected confidence threshold that was frozen before test evaluation.
- RT-DETR speed was reported by the Ultralytics validation pipeline, while RF-DETR speed used a custom batch-size-1 benchmark. Speed values are therefore implementation-dependent.
- Some source annotations contain mistakes, including non-plate objects labelled as plates.
- The emirate dataset is strongly imbalanced toward Dubai.
- Umm Al Quwain has few test examples, so its class-level result is less statistically reliable.
- OCR quality can decrease when detector crops are loose, shifted, blurred, or incomplete.

## Team Handoff

A one-page starting guide is provided in:

```text
START_HERE.md
```

The complete contribution and report-writing handoff is available in:

```text
docs/FINAL_TEAM_HANDOFF.md
```
