# CPE393_TBA_MLOps

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Directory Structure

The repository is structured as follows:

```
├── data/
│   └── adult.csv
├── models/
├── notebooks/
│   ├── 01_Analyze_and_Visualization.ipynb
│   └── 02_compare-all-the-classification-models.ipynb
├── scripts/
│   └── train.py
├── requirements.txt
└── README.md
```

## Usage

(

    not available yet, but you can use conda to create the environment and activate it.

    ```
    conda env create -f environment.yml
    conda activate mlops-project
    ```

    Docker usage : 

    docker build -t ml-project .

    docker run -p 8888:8888 -v $(pwd):/app ml-project

    and you can access the jupyter notebook at http://localhost:8888/

)


### Setting Up the Environment

1. Clone the repository:

```bash
git clone https://github.com/chrnthnkmutt/CPE393_TBA_MLOps.git
cd CPE393_TBA_MLOps
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Scripts

To run the training script, use the following command:

```bash
python scripts/train.py
```

### Using the Notebooks

The Jupyter notebooks are located in the `notebooks/` directory. You can open and run them using Jupyter Notebook or JupyterLab.

```bash
jupyter notebook notebooks/01_Analyze_and_Visualization.ipynb
jupyter notebook notebooks/02_compare-all-the-classification-models.ipynb
```

## Additional Information

This repository is structured to follow best practices for Machine Learning Operations (MLOps). The directory structure separates different functionalities such as data processing, model training, and evaluation. This helps in maintaining a clean and organized codebase, making it easier to manage and scale machine learning projects.

## Checklist
✅   Training Data Management

_   Containerize with Docker (Romain)

✅   Data Cleansing and Feature Engineering

✅   Building Data Pipeline

✅   Hyperparameter Tuning

_   Model Development and Experiment Tracking (MLFlow - Tee) - Wednesday Night

_   Model Deployment and Flask (Romain)

_   Model Mornitoring (Japan, Boat, Tee) - Tomorrow