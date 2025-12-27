import streamlit as st
import pandas as pd
import joblib

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Apple Price Predictor",
    page_icon="apple_logo.png",
    layout="centered"
)

# ===============================
# LOAD MODEL
# ===============================
model = joblib.load("apple_price_model.pkl")

# ===============================
# HEADER
# ===============================
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:15px;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/f/fa/Apple_logo_black.svg" width="50">
        <h1 style="margin:0;">Apple Product Price Predictor</h1>
    </div>

    <p style="color:gray; margin-top:4px;">
        NLP Project · Price Prediction
    </p>

    <p style="font-size:14px; color:#555;">
        By: <b>2023-BS-AI-102</b>, <b>2023-BS-AI-166</b>, <b>2025S-BS-AI-034</b>
    </p>
    """,
    unsafe_allow_html=True
)
st.markdown("---")

# ===============================
# INPUT SECTION
# ===============================
st.markdown("### 🔧 Product Configuration")

col1, col2 = st.columns(2)

with col1:
    category = st.selectbox(
        "Product Category",
        ["iPhone", "iPad", "Mac", "Watch"]
    )

    product = st.text_input(
        "Product Name",
        "Enter product name"
    )

    # STORAGE (conditional)
    if category == "Watch":
        storage_gb = 0
        st.info("Storage not applicable for Watch")
    else:
        storage = st.selectbox(
            "Storage",
            ["64 GB", "128 GB", "256 GB", "512 GB", "1 TB", "2 TB"]
        )

with col2:
    color = st.text_input(
        "Color",
        "Enter required color"
    )

    # CHIP (conditional)
    if category in ["iPhone", "iPad"]:
        chip = st.selectbox("Chip", ["A14", "A15", "A16", "A17"])
    elif category == "Mac":
        chip = st.selectbox("Chip", ["M1", "M2", "M3"])
    else:
        chip = ""

# ===============================
# STORAGE PARSING
# ===============================
if category != "Watch":
    def parse_storage(storage):
        if "TB" in storage:
            return int(storage.replace(" TB", "")) * 1024
        return int(storage.replace(" GB", ""))

    storage_gb = parse_storage(storage)

st.markdown("---")

# ===============================
# PREDICTION
# ===============================
if st.button("💰 Predict Price", use_container_width=True):

    # CATEGORY–PRODUCT VALIDATION
    if category.lower() not in product.lower():
        st.warning(
            f"Product name does not match selected category ({category}). "
            "Please correct it."
        )
        st.stop()

    spec_text = f"{product} {chip} {storage_gb}GB {color}"

    input_df = pd.DataFrame([{
        "Category": category,
        "Spec_Text": spec_text,
        "Storage_GB": storage_gb
    }])

    price = model.predict(input_df)[0]

    st.markdown(
        f"""
        <div style="
            background-color:#f5f5f7;
            padding:25px;
            border-radius:16px;
            text-align:center;
            margin-top:20px;
        ">
            <h3 style="margin-bottom:6px;">Predicted Price</h3>
            <h1 style="margin:0;">Rs. {int(price):,}</h1>
            <p style="color:gray; margin-top:8px;">
                Prediction generated using a trained ML model
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ===============================
# FOOTER
# ===============================
st.markdown(
    """
    <hr>
    <p style="text-align:center; color:gray; font-size:13px;">
        NLP Project · Predictions are based on learned patterns from scraped data
    </p>
    """,
    unsafe_allow_html=True
)
