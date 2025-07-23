# Scalable Medical Data Curation and Labelling

This repository provides a **demonstrator** for an end-to-end curation and labelling pipeline of medical image datasets **at scale**—demonstrated here for prostate cancer detection on biparametric MRI from internal data sources at **Brigham and Women's Hospital, Boston**. The goal is to facilitate the curation of labeled datasets, making them more accessible and accelerating the development and **clinical adoption** of **NeuralNet-based algorithms**.

## Repo Overview

The pipeline leverages automated extraction, retrieval, and matching of clinical entities from thousands of radiology and pathology reports, as well as biopsy coordinates, to generate **weak labels (bounding boxes)** for training deep learning models. **Key features include:** automated entity extraction from unstructured clinical text using LLMs and NLP, cross-source data matching and programmatic label generation, tools for dataset export and downstream model experimentation.


### Installation

This project uses [uv](https://github.com/astral-sh/uv) for Python package management.

```sh
# Install uv (if not already installed)
pip install uv

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Repository Structure

- `main.py` — Pipeline entry point
- `preprocessing.py` — Clinical text preprocessing and parsing
- `extraction.py` — LLM-based entity extraction from reports  
- `matching.py` — Cross-source entity matching logic
- `postprocessing.py` — Label generation and validation
- `export.py` — Dataset export utilities
- `llm.py` — Language model interaction utilities
- `configs.yaml` — Pipeline configuration parameters
- `cv_model_experiments/` — Example downstream model training scripts
- `synthetic_output_dfs/` — Synthetic demonstration data
- `media/` — Figures and thesis PDF for documentation

### Demonstration Data

Since raw clinical reports are rarely available in public datasets, this repository includes **synthetic demonstration data** in `synthetic_output_dfs/` that shows the expected pipeline outputs for the original prostate cancer detection use case (see below):

- `clinical_metadata_10_synthetic_samples.csv` - Sample patient metadata with extracted clinical entities 
- `image_acquisition_10_synthetic_samples.csv` - Sample imaging parameters and scanner information

**Note**: These are completely synthetic examples created for demonstration purposes only.

## Motivation

**Labeled medical imaging data is scarce, and expert annotation is not scalable.** At the time of developing this work, the lack of large-scale labeled datasets is a major bottleneck preventing neural networks from becoming clinically relevant in diagnostic tasks. Traditional approaches rely on costly human experts to manually annotate medical images, which is:

- **Prohibitively expensive & time-consuming** at the scale needed for deep learning
- **Subjective & limited by ground truth** with at times substantial inter-observer variability and lack of histopathological confirmation (e.g. in prostate cancer detection case)

**Our solution**: Build an automated system that can create labeled datasets with **no human supervision** by intelligently mining the vast amounts of unstructured clinical data already available in hospital systems.

## Original Use Case: Prostate Cancer Detection and Grading on Biparametric MRI

Instead of relying on **costly human-experts to annotate tumors** on images (often with no ground truth in terms of histopathologically-confirmed tumors) we constructed bounding boxes as "weak labels" by mining **>12k Radiology reports**, **>4k Pathology reports** and **>600 Biopsy Target coordinates**. We created a large, labeled dataset, suitable for training deep learning models without manual expert annotation.

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

This automated approach allowed for the creation of large-scale datasets that would be prohibitively expensive and time-consuming to create manually. The goal was to use these "weak" labels to increase dataset sizes for training and fine-tuning of computer vision models for automated cancer detection.

## Important Limitations & Why This Work Might Still Matter

- **Dataset specificity & limited generalizability**: This pipeline was designed for **internal data at Brigham and Women's Hospital**, with specific formatting and may not generalize directly to public datasets.
- **Lack of public benchmarks**: Requires raw clinical reports for benchmarking and testing (currently almost non-existent in public datasets) 
- **The field has evolved**: Joint multi-modal embedding approaches, representation learning, and foundation models now often outperform explicit entity extraction and label construction methods

Despite these limitations, this repository provides value as:

- **Interpretable approach**: Explicit entity extraction may be more interpretable for humans than pure embedding approaches
- **Resource-constrained settings**: Where massive multimodal medical datasets are scarce, simple data curation pipelines like this can help address the cold-start problem
- **Systematic methodology & baseline**: Provides a structured approach to clinical text mining and cross-source data matching that serves as a useful comparison point

## Future Directions

- **Prospective medical data curation**: The most promising path may be integrating AI development into clinical workflows from the outset—deploying lightweight curation pipelines like this during data generation to ensure patient care contributes systematically to improving treatment through continuous AI training
- **Joint embedding approaches**: Move toward text-image joint representation learning
- **Cross-institutional validation**: Test generalizability across different healthcare systems

## Acknowledgements & Contributing

This work was developed at Brigham and Women's Hospital as part of research into automated medical dataset curation. While the specific implementation may not generalize to all settings, we hope the systematic approach to clinical text mining and entity extraction provides a useful baseline for the research community. Contributions are always welcome!

---

*This automated approach represents one systematic method for clinical data curation, addressing the critical need for labeled medical imaging data that existed at the time of development.*