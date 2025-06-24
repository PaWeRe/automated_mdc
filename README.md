# Scalable Medical Data Curation and Labelling

This repository provides an **end-to-end retrieval pipeline** for curating labelled medical image datasets **at scale**—demonstrated here for prostate cancer detection on biparametric MRI. **The goal** is to facilitate the curation of labeled datasets, making them more accessible and accelerating the development and **clinical adoption of neural network-based algorithms**.


## Repo Overview

The pipeline leverages automated extraction, retrieval, and matching of clinical entities from thousands of radiology and pathology reports, as well as biopsy coordinates, to generate **weak labels (bounding boxes)** for training deep learning models. 

**Key features:**
- Automated entity extraction from unstructured clinical text using LLMs and NLP
- Cross-source data matching and programmatic label generation
- Tools for dataset export and downstream model experimentation

## Folder Structure

- `main.py`                — Entry point for running the pipeline
- `preprocessing.py`       — Preprocessing routines for clinical text and imaging data
- `extraction.py`          — LLM-based and rule-based entity extraction from reports
- `matching.py`            — Logic for matching entities across data sources
- `postprocessing.py`      — Postprocessing and label generation
- `export.py`              — Export and dataset creation utilities
- `llm.py`                 — LLM interaction utilities
- `configs.yaml`           — Configuration file for pipeline parameters
- `requirements.txt`       — Python dependencies
- `cv_model_experiments/`  — Example scripts for model training, inference, and evaluation 
    - `training.py`, `inference.py`, `model_eval.py`
- `anonymized_output_dfs/` — Example anonymized output dataframes (CSV)
    - `clinical_metadata_v1_anonymized.csv`, `image_acquisition_params_v1_anonymized.csv`
- `media/`                 — Figures and thesis PDF for documentation
    - Pipeline diagrams, example outputs, thesis PDF

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for Python package management and virtual environments.

1. **Install uv** (if not already installed):
   ```sh
   pip install uv
   ```
2. **Create a virtual environment and install dependencies:**
   ```sh
   uv venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```

## Example: Constructing labels for prostate cancer detection and grading on biparametric MRI

Instead of relying on **costly human-experts to annotate tumors** on images (often with no ground truth in terms of histopathologically-confirmed tumors) we construct bounding boxes as "weak labels" by mining **>12k Radiology reports**, **>4k Pathology reports** and **>600 Biopsy Target coordinates**. By leveraging a vast amount of unstructured clinical data, we can create a large, labeled dataset suitable for training deep learning models without manual expert annotation.

### Data Curation Overview
The overall process involves several stages, from raw data ingestion to the final generation of labeled image datasets. The following diagram provides a high-level overview of the pipeline.

![Data Curation Overview](./media/data_curation_overview.png)

### The Pipeline in Detail

#### 1) Data Input
The pipeline starts with **four** primary sources of clinical data:
-   **Radiology Reports (>12,000):** Unstructured text reports from MRI scans, describing findings.
-   **Pathology Reports (>4,000):** Reports detailing histopathological findings from biopsy samples.
-   **Biopsy Target Coordinates (>600):** Semi-structured data indicating the precise location of biopsies.
-   **Imaging studies (>1,900):** Imaging studies containing >30k image files with significantly varying Series descriptions

![Data Input examples](./media/data_input.png)

#### 2) Automated Curation Process
The core of the project is the automated curation process, which includes:
-   **LLM-based Entity Extraction:** We use LLMs pre-trained on clinical text, from HF's Transformer library to extract key clinical entities from Radiology and Pathology reports (e.g., PI-RADS scores, tumor locations, Gleason scores).
-   **Retrieval and Matching:** The extracted entities are then used to retrieve and match data across the different sources. For example, a patient's radiology report is matched with their pathology results and biopsy coordinates.
-   **Label Generation:** Finally, the matched and verified data is used to export bounding boxes as segmentations in DICOM or nifti, as well as the three imaging modalities usually used for prostate cancer diagnosis (T2, DWI, ADC).

![Detailed Curation Process](./media/data_curation_process.png)

#### 3) Pipeline Output
The output of the pipeline is a curated dataset of biparametric MRI scans with bounding box annotations for prostate cancer lesions. The bounding boxes are approximations of expert-derived tumor segmentations and use the coordinate of the MR guided biopsy as reference point and the lesion measurements described in the radiology reports as radius.

Here are some examples of the generated labels:

![Example Output 1](./media/data_output.png)
![Example Output 2](./media/data_output_2.png)
![Example Output 3](./media/data_output_3.png)

This automated approach allows for the creation of large-scale datasets that would be prohibitively expensive and time-consuming to create manually. The hope is to use these "weak" lables to increase dataset sizes for training and fine-tuning of computer vision models for automated cancer detection (performance gains and model benchmarking TBD!).