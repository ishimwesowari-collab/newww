# app.py
import os
import traceback
import pickle
import numpy as np
import streamlit as st

# try to import joblib but continue if it's not installed
try:
    import joblib
    HAVE_JOBLIB = True
except Exception:
    HAVE_JOBLIB = False

MODEL_PATH = "25RP18587.sav"  # change to .pkl or .joblib if your file differs

st.title("🌾 Crop Yield Prediction App")
st.write("Enter the temperature value to predict the crop yield.")

def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")
    # prefer joblib when available (more robust for sklearn objects)
    if HAVE_JOBLIB:
        try:
            return joblib.load(path)
        except Exception:
            # fall through to pickle
            pass
    # fallback to pickle
    with open(path, "rb") as f:
        return pickle.load(f)

# Load model with friendly errors shown in the Streamlit app
try:
    model = load_model(MODEL_PATH)
except FileNotFoundError as e:
    st.error(str(e))
    st.info("Put the model file in the same repo/folder as this app and redeploy. Update MODEL_PATH if the filename or path is different.")
    st.stop()
except ModuleNotFoundError as e:
    # e.g. No module named 'sklearn'
    st.error(f"Missing package required to unpickle the model: {e}")
    st.info("Add the missing package(s) to requirements.txt (e.g. scikit-learn) and redeploy.")
    st.stop()
except Exception:
    st.error("An error occurred while loading the model. See traceback below.")
    st.text(traceback.format_exc())
    st.stop()

# User input for temperature
temperature = st.number_input(
    "Enter the temperature (°C)",
    min_value=-50.0,
    max_value=100.0,
    step=0.1,
    value=27.0
)

# Prediction button
if st.button("Predict Yield"):
    try:
        X = np.array([[float(temperature)]])  # 2D array for model
        pred = model.predict(X)
        st.success(f"🌱 Predicted Crop Yield: {float(pred[0]):.2f} units")
    except Exception:
        st.error("Prediction failed — see traceback below.")
        st.text(traceback.format_exc())
