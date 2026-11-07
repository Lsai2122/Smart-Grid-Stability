from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
from lime.lime_tabular import LimeTabularExplainer
from streamlit.runtime.scriptrunner import get_script_run_ctx

if get_script_run_ctx() is None:
    print('This is a Streamlit app. Run it with:')
    print(r'streamlit run d:\Desktop\ML Research Papers\project\streamlit_app.py')
    sys.exit(0)

st.set_page_config(page_title='Smart Grid Stability Predictor', page_icon='⚡', layout='wide')
st.title('Smart Grid Stability Predictor')

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / 'models'
DATA_PATH = BASE_DIR / 'smart_grid_stability_augmented.csv'


@st.cache_resource
def load_artifacts():
    mlp_artifact = joblib.load(MODELS_DIR / 'mlp_model.joblib')
    xgb_artifact = joblib.load(MODELS_DIR / 'xgboost_model.joblib')

    feature_cols = mlp_artifact['feature_cols']
    warmup_row = pd.DataFrame([[0.0] * len(feature_cols)], columns=feature_cols)
    _ = mlp_artifact['model'].predict_proba(warmup_row)
    _ = xgb_artifact['model'].predict_proba(warmup_row)

    return {'MLP': mlp_artifact, 'XGBoost': xgb_artifact}


def label_from_prob(prob_unstable: float) -> str:
    return 'Unstable' if prob_unstable >= 0.5 else 'Stable'


def get_background_df(feature_cols, artifact):
    bg = artifact.get('background_data')
    if bg:
        return pd.DataFrame(bg, columns=feature_cols)

    df = pd.read_csv(DATA_PATH)
    return df[feature_cols].sample(n=min(200, len(df)), random_state=42).copy()


def model_predict_proba_df(model, feature_cols, arr_or_df):
    if isinstance(arr_or_df, pd.DataFrame):
        return model.predict_proba(arr_or_df)
    return model.predict_proba(pd.DataFrame(arr_or_df, columns=feature_cols))


def extract_class1_shap_row(shap_values):
    if isinstance(shap_values, list):
        idx = 1 if len(shap_values) > 1 else 0
        return np.array(shap_values[idx])[0]

    arr = np.array(shap_values)
    if arr.ndim == 3:
        class_idx = 1 if arr.shape[-1] > 1 else 0
        return arr[0, :, class_idx]
    if arr.ndim == 2:
        return arr[0]
    return arr.reshape(-1)


def top_feature_rows(values_row, feature_cols, top_k=6):
    order = np.argsort(np.abs(values_row))[::-1][:top_k]
    return [{'feature': feature_cols[i], 'contribution': float(values_row[i])} for i in order]


def run_single_prediction(model_name, model, feature_cols, input_df):
    # Measure only the model inference call (exclude UI/status waiting time).
    t0 = time.perf_counter()
    prob_unstable = float(model.predict_proba(input_df)[0, 1])
    model_compute_ms = (time.perf_counter() - t0) * 1000.0

    return {
        'prob_unstable': prob_unstable,
        'label': label_from_prob(prob_unstable),
        'prediction_ms': model_compute_ms,
        'lime': None,
        'shap': None,
    }


try:
    artifacts = load_artifacts()
except FileNotFoundError:
    st.error('Model files are missing. Run train_mlp.py and train_xgboost.py first.')
    st.stop()

feature_cols = artifacts['MLP']['feature_cols']
background_dfs = {
    'MLP': get_background_df(feature_cols, artifacts['MLP']),
    'XGBoost': get_background_df(feature_cols, artifacts['XGBoost']),
}


@st.cache_resource
def load_explainers():
    explainers = {}

    for model_name in ['MLP', 'XGBoost']:
        model = artifacts[model_name]['model']
        background_df = background_dfs[model_name]

        lime_explainer = LimeTabularExplainer(
            training_data=np.array(background_df),
            feature_names=feature_cols,
            class_names=['stable', 'unstable'],
            mode='classification',
            random_state=42,
        )

        if model_name == 'XGBoost':
            shap_explainer = shap.TreeExplainer(model)
        else:
            shap_explainer = shap.KernelExplainer(
                lambda arr, m=model: model_predict_proba_df(m, feature_cols, arr),
                background_df,
            )

        explainers[model_name] = {
            'lime': lime_explainer,
            'shap': shap_explainer,
        }

    return explainers


explainers = load_explainers()

st.subheader('Input Features')
defaults = {
    'tau1': 3.0, 'tau2': 3.0, 'tau3': 8.0, 'tau4': 9.5,
    'p1': 3.8, 'p2': -0.8, 'p3': -1.3, 'p4': -1.7,
    'g1': 0.65, 'g2': 0.86, 'g3': 0.88, 'g4': 0.95,
}

input_vals = {}
cols = st.columns(4)
for i, feat in enumerate(feature_cols):
    with cols[i % 4]:
        input_vals[feat] = st.number_input(feat, value=float(defaults.get(feat, 0.0)), format='%.6f')


if st.button('Predict Stability (MLP -> XGBoost)', type='primary'):
    input_df = pd.DataFrame([input_vals], columns=feature_cols)
    status = st.empty()

    mlp_result = run_single_prediction('MLP', artifacts['MLP']['model'], feature_cols, input_df)

    xgb_result = run_single_prediction('XGBoost', artifacts['XGBoost']['model'], feature_cols, input_df)


    st.session_state['prediction_result'] = {
        'input': input_vals,
        'models': {
            'MLP': mlp_result,
            'XGBoost': xgb_result,
        },
    }


if 'prediction_result' in st.session_state:
    result = st.session_state['prediction_result']
    input_df = pd.DataFrame([result['input']], columns=feature_cols)

    st.subheader('Predictions')
    c1, c2 = st.columns(2)

    for col, model_name in [(c1, 'MLP'), (c2, 'XGBoost')]:
        model_data = result['models'][model_name]
        model = artifacts[model_name]['model']
        background_df = background_dfs[model_name]
        lime_explainer = explainers[model_name]['lime']
        shap_explainer = explainers[model_name]['shap']

        with col:
            st.markdown(f'### {model_name}')
            st.metric('Prediction', model_data['label'])
            st.metric('Unstable Probability', f"{model_data['prob_unstable']:.4f}")
            st.metric('Prediction Time', f"{model_data['prediction_ms']:.4f} ms")

            if model_data['label'] == 'Unstable' and model_data['lime'] is None:
                t0 = time.perf_counter()
                exp = lime_explainer.explain_instance(
                    data_row=input_df.iloc[0].values,
                    predict_fn=lambda arr, m=model: model_predict_proba_df(m, feature_cols, arr),
                    num_features=6,
                )
                lime_ms = (time.perf_counter() - t0) * 1000.0

                model_data['lime'] = {
                    'time_ms': lime_ms,
                    'rows': [{'feature_rule': term, 'weight': float(weight)} for term, weight in exp.as_list()],
                }

            # For unstable predictions, generate SHAP automatically after LIME (sequential, no async).
            if model_data['label'] == 'Unstable' and model_data['shap'] is None:
                t0 = time.perf_counter()
                if model_name == 'XGBoost':
                    shap_values = shap_explainer.shap_values(input_df)
                else:
                    shap_values = shap_explainer.shap_values(input_df, nsamples=100)

                shap_row = extract_class1_shap_row(shap_values)
                shap_ms = (time.perf_counter() - t0) * 1000.0
                model_data['shap'] = {
                    'time_ms': shap_ms,
                    'rows': top_feature_rows(shap_row, feature_cols, top_k=6),
                }

            if model_data['label'] == 'Unstable':
                st.markdown('**LIME Explanations**')
                st.dataframe(pd.DataFrame(model_data['lime']['rows']), use_container_width=True)
                st.caption(f"LIME explanation time: {model_data['lime']['time_ms']:.4f} ms")
                st.markdown('**SHAP Explanations**')
                st.dataframe(pd.DataFrame(model_data['shap']['rows']), use_container_width=True)
                st.caption(f"SHAP explanation time: {model_data['shap']['time_ms']:.4f} ms")
            else:
                st.info('Prediction is Stable, so LIME/SHAP are not auto-generated.')

            lime_ms = model_data['lime']['time_ms'] if model_data['lime'] is not None else 0.0
            shap_ms = model_data['shap']['time_ms'] if model_data['shap'] is not None else 0.0
            st.caption(
                f"Total (Prediction + LIME): {model_data['prediction_ms'] + lime_ms:.4f} ms | "
                f"Total (Prediction + SHAP): {model_data['prediction_ms'] + shap_ms:.4f} ms"
            )
