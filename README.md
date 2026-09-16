# UoM-SurgicalAI — ORena SAVE FOCUS Challenge

This repository contains the code used by **UoM-SurgicalAI** for submissions to the **ORena SAVE FOCUS Challenge at MICCAI 2026**.

The repository contains two main components

- Docker inference pipelines for the FRAME, SEGMENT, and PROCEDURE tracks
- Data preparation, DoRA fine-tuning, prediction, and evaluation code for adapting InternVL3.5-8B-Instruct to ORena FOCUS surgical VQA

## Repository Structure

```text
ORENA2026/
├── orena-docker/
│   ├── frame-algorithm/
│   ├── procedure-algorithm/
│   ├── segment-algorithm/
│   └── segment-algorithm-v1/
│
└── dora-finetune/
    ├── build_orena_segment_dora_jsonl.py
    ├── prepare_orena_segment_dora_data.py
    ├── dora_fine_tune.py
    ├── run_dora_prediction.py
    └── eval_segment_predictions.py
```

## Docker Submission Code

The `orena-docker/` directory contains the inference code and Docker configurations used to prepare challenge submissions.

Each track directory contains the code required to build and run the corresponding challenge container.

### FRAME Track

```text
orena-docker/frame-algorithm/
```

Contains the Docker inference pipeline used for the FRAME Track submission.

### SEGMENT Track

```text
orena-docker/segment-algorithm/
```

Contains the Docker inference pipeline used for the SEGMENT Track submission.

### PROCEDURE Track

```text
orena-docker/procedure-algorithm/
```

Contains the final PROCEDURE Track inference pipeline.

The method uses **InternVL3.5-8B-Instruct** with a **DoRA-adapted language model** and question-conditioned temporal frame sampling for long procedure-level videos.

The overall inference pipeline is

```text
Procedure-level video + question
        ↓
Temporal intent routing
        ↓
Question-conditioned frame sampling
        ↓
Timestamped multimodal prompt
        ↓
InternVL3.5-8B-Instruct + DoRA adapter
        ↓
Raw model response
        ↓
Answer normalization
        ↓
Final VQA answer
```

The final submission uses the **epoch-5 DoRA adapter**.

## DoRA Fine-Tuning

The `dora-finetune/` directory contains the data preparation, training, prediction, and evaluation code used for parameter-efficient adaptation of **InternVL3.5-8B-Instruct**.

The DoRA adapter used in the final model was trained on the official ORena FOCUS **SEGMENT training split**. No PROCEDURE VQAs were used for parameter updates.

## Fine-Tuning Scripts

### `prepare_orena_segment_dora_data.py`

Prepares the visual data used for ORena SEGMENT supervised fine-tuning.

### `build_orena_segment_dora_jsonl.py`

Builds the SFT JSONL files used by the DoRA training pipeline.

### `dora_fine_tune.py`

Fine-tunes InternVL3.5-8B-Instruct using DoRA.

The training script

- loads the pretrained InternVL3.5-8B-Instruct backbone
- freezes the vision encoder, multimodal projector, and base language-model weights
- attaches DoRA to the language-model attention projections
- performs multimodal supervised fine-tuning
- evaluates validation cross-entropy loss
- reduces the learning rate when validation loss stops improving
- saves per-epoch, best, and final checkpoints

### `run_dora_prediction.py`

Runs prediction using the fine-tuned model for local evaluation.

### `eval_segment_predictions.py`

Evaluates SEGMENT predictions produced by the fine-tuned model.

## Docker Build and Test

Each algorithm directory contains the files required to build and test the corresponding Docker image.

Typical files include

```text
Dockerfile
requirements.txt
do_build.sh
do_test_run.sh
do_save.sh
inference.py
```

For example, the PROCEDURE image can be built and tested with

```bash
cd orena-docker/procedure-algorithm

./do_build.sh
./do_test_run.sh
./do_save.sh
```


## Datasets

The methods in this repository use the official ORena FOCUS challenge datasets.

### HeiCo-FOCUS-VQA

https://huggingface.co/datasets/orena-dkfz/heico-focus-vqa

### LapChole-FOCUS-VQA

https://huggingface.co/datasets/orena-dkfz/lapchole-focus-vqa

The datasets and videos are **not redistributed in this repository**. Please obtain them through the official ORena FOCUS release and follow the corresponding licenses and data usage agreements.

## Model Weights and Checkpoints

Here are the pretrained InternVL3.5-8B-Instruct weights and trained DoRA checkpoint on epoch 5: https://drive.google.com/drive/folders/1wV50zkEBHCGOlZuu7pvl7nfey74nDor8?usp=sharing

For local Docker builds, the required resources should be placed under the corresponding algorithm `resources/` directory.

For example

```text
procedure-algorithm/
└── resources/
    ├── InternVL3_5-8B-Instruct/
    └── checkpoint-epoch-5/
        └── dora_adapter/
```


## Challenge Information

**Team**  
UoM-SurgicalAI

**Challenge**  
ORena SAVE FOCUS Challenge

**Conference**  
MICCAI 2026

**Tracks**  
FRAME, SEGMENT, PROCEDURE

**Challenge Website**  
https://orena-focus-challenge.org/

**ORena Website**  
https://or-arena.org/

## Acknowledgements

We thank the ORena SAVE FOCUS Challenge organizers for providing the challenge datasets, evaluation framework, and submission infrastructure.

This repository was developed by **UoM-SurgicalAI** for participation in the ORena SAVE FOCUS Challenge at MICCAI 2026.