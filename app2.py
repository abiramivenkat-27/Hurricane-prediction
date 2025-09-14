import streamlit as st
import pandas as pd
import joblib
import json

# -----------------------------
# MUST be the first Streamlit command
# -----------------------------
st.set_page_config(page_title="Hurricane Track Prediction", layout="centered")

# -----------------------------
# Load Model & Metadata
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("final_hurricane_model.pkl")
    with open("model_metadata.json", "r") as f:
        metadata = json.load(f)
    return model, metadata

model, metadata = load_model()
top_features = metadata["features"]

# -----------------------------
# Streamlit App Layout
# -----------------------------
st.title("🌪️ Hurricane Track Prediction App")
st.markdown(
    "This app predicts **hurricane track deltas (latitude & longitude changes)** "
    "using the trained Random Forest model."
)

# -----------------------------
# Mode Selection
# -----------------------------
mode = st.radio("Choose Prediction Mode:", ["Single Prediction", "Batch Prediction (CSV Upload)"])

# -----------------------------
# Single Input Prediction
# -----------------------------
if mode == "Single Prediction":
    st.subheader("Enter Feature Values")

    input_data = {}
    for feature in top_features:
        input_data[feature] = st.number_input(f"{feature}", value=0.0, format="%.4f")

    input_df = pd.DataFrame([input_data])

    if st.button("Predict Track Δ"):
        prediction = model.predict(input_df)
        delta_lat, delta_lon = prediction[0]

        st.success(f"✅ Predicted Δ Latitude: {delta_lat:.4f}")
        st.success(f"✅ Predicted Δ Longitude: {delta_lon:.4f}")

# -----------------------------
# Batch Prediction (CSV Upload)
# -----------------------------
else:
    st.subheader("Upload CSV File for Batch Prediction")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        # Load CSV
        df_input = pd.read_csv(uploaded_file)
        st.success("📂 File uploaded successfully!")
        st.write("Preview of uploaded data:")
        st.dataframe(df_input.head())

        # Check required features
        missing = [f for f in top_features if f not in df_input.columns]
        if missing:
            st.error(f"❌ Missing required features in CSV: {missing}")
        else:
            try:
                # Run prediction
                preds = model.predict(df_input[top_features])
                df_input["pred_delta_lat"] = preds[:, 0]
                df_input["pred_delta_lon"] = preds[:, 1]

                st.subheader("✅ Prediction Results (first 10 rows)")
                st.dataframe(df_input.head(10))

                # Allow download
                csv = df_input.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Predictions as CSV",
                    data=csv,
                    file_name="hurricane_predictions.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"⚠️ Error while predicting: {e}")
