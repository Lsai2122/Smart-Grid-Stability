# Smart Grid Stability Prediction

A machine learning project for classifying simulated power grid operating conditions as **stable** or **unstable**. It includes an exploratory and comparative notebook, dedicated model training notebooks, saved model artifacts, and a Streamlit app for trying predictions with local explanations.

> This is an educational project using a prepared dataset. Predictions are not validated for operational use in a real power grid.

## Project at a glance

The project uses 12 input features describing four grid nodes:

- `tau1`–`tau4`: node time constants
- `p1`–`p4`: active power values
- `g1`–`g4`: price coefficients

The source labels are `stab` (a continuous stability measure) and `stabf` (the categorical stability class). The training notebooks convert `stabf` to a binary target: `stable` = 0 and `unstable` = 1. They use a stratified 85/15 train/test split with a fixed random seed and report accuracy, precision, recall, F1 score, and ROC-AUC.

## Models

Two classifiers are trained and used by the app:

- **Multi-Layer Perceptron (MLP):** a scikit-learn pipeline with `StandardScaler` and an `MLPClassifier` using hidden layers `(256, 128)`, early stopping, and up to 350 iterations.
- **XGBoost:** an `XGBClassifier` with 300 trees, maximum depth 8, and learning rate 0.05.

The notebooks save the trained estimators, feature names, evaluation metrics, class names, and background samples as Joblib artifacts in `models/`. The app uses LIME and SHAP to explain predictions classified as unstable. SHAP uses `TreeExplainer` for XGBoost and `KernelExplainer` for the MLP.

## Repository contents

```text
.
├── Main.ipynb                              # Project exploration, comparison, and explainability
├── train_mlp.ipynb                         # Train and export the MLP model
├── train_xgboost.ipynb                     # Train and export the XGBoost model
├── streamlit_app.py                        # Interactive prediction app
├── smart_grid_stability_augmented.csv      # Prepared dataset used by notebooks and app
├── models/
│   ├── mlp_model.joblib                    # Saved MLP pipeline and metadata
│   └── xgboost_model.joblib                 # Saved XGBoost model and metadata
└── Group_4_Smart_Grid_Prediction.pptx      # Project presentation
```

## Setup

Use Python 3.10 or newer. From the repository root, create and activate a virtual environment, then install the project dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
source .venv/bin/activate
```

Install the packages used by the app and notebooks:

```bash
python -m pip install --upgrade pip
python -m pip install jupyter streamlit numpy pandas scikit-learn xgboost joblib lime shap matplotlib seaborn
```

## Run the app

The checked-in model files and dataset are expected to remain in their repository locations. Start the Streamlit app from the repository root:

```bash
streamlit run streamlit_app.py
```

Enter values for the 12 grid features and select **Predict Stability (MLP -> XGBoost)**. The app displays each model's class, unstable probability, and prediction time. For unstable predictions, it also computes local LIME and SHAP feature attributions.

## Retrain the models

Open `train_mlp.ipynb` or `train_xgboost.ipynb` in Jupyter and run its cells from top to bottom. Each notebook reads `smart_grid_stability_augmented.csv`, creates the binary target from `stabf`, evaluates the model, and writes the corresponding artifact under `models/`.

`Main.ipynb` contains the broader analysis, model comparison, and explainability work. Run it in Jupyter to explore the project workflow and analysis.

## Data and artifacts

The CSV is included so the notebooks and app can run without a separate data download. The saved Joblib files are included so the app can run without first retraining. Re-running a training notebook replaces that model's artifact.

Joblib model files are pickle-based artifacts. Only load artifacts from sources you trust.

## Notes

- Run commands from the repository root so the app and notebooks can find the CSV and model files at their relative paths.
- Model metrics depend on the checked-in data, package versions, and training run; use the notebooks to reproduce the reported evaluation for a particular environment.
- The dataset and saved models are project assets and may make a fresh clone larger than a code-only repository.
