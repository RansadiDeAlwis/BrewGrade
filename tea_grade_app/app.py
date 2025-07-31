import streamlit as st
import numpy as np
import joblib
import base64
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# === Load Models ===
model_svm = joblib.load("tea_grade_app/models/tea_model.pkl")
label_encoder = joblib.load("tea_grade_app/models/label_encoder.pkl")
cnn_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# === Feature Extraction ===
def extract_features(img_pil):
    img = img_pil.resize((224, 224))
    img = np.array(img)
    if img.shape[-1] == 4:
        img = img[:, :, :3]  # remove alpha channel
    img = preprocess_input(img.astype(np.float32))
    features = cnn_model.predict(np.expand_dims(img, axis=0), verbose=0)
    return features.flatten()

# === Predict ===
def predict_image(img_pil):
    features = extract_features(img_pil)
    pred_encoded = model_svm.predict([features])
    pred_label = label_encoder.inverse_transform(pred_encoded)
    return pred_label[0]

# === Page Config ===
st.set_page_config(page_title="BrewGrade", page_icon="🍃", layout="wide")

# === Inject Custom Fonts ===
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=CoFo+Sans&family=Forum&family=Lato&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Lato', sans-serif;
    }
    h1, h2, h3, h4, h5 {
        font-family: 'Forum', serif;
    }
    .stButton>button {
        font-family: 'CoFo Sans', sans-serif;
    }
    .stSelectbox>div>div>div {
        font-family: 'CoFo Sans', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# === Sidebar Navigation ===
st.sidebar.image("tea_grade_app/assests/home.jpg", use_container_width=True)
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Choose Page", ["Home", "About", "Tea Grading"])

# === Home Page ===
if app_mode == "Home":
    st.markdown(
        """
        <div style="
            max-width: 1100px;
            margin: 2rem auto 3rem auto;
            padding: 2rem;
            background: linear-gradient(135deg, #a8d5a3 0%, #5a9e44 100%);
            color: white;
            text-align: center;
            border-radius: 15px;
            box-shadow: 0 8px 20px rgba(0, 100, 20, 0.3);
            font-family: 'Lato', sans-serif;
        ">
            <h1 style="font-family: 'Forum', serif; font-weight: 700; font-size: 3rem; margin-bottom: 0.2rem;">
                🍃 Welcome to BrewGrade 🍃
            </h1>
            <p style="font-size: 1.25rem; margin-top:0; font-style: italic;">
                AI-powered instant tea leaf grade prediction
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    original_image = Image.open("tea_grade_app/assests/home_page.jpg")
    st.markdown(
        f"""
        <div style="width: 100%; margin: 0 auto; overflow: hidden;">
            <img src="data:image/jpeg;base64,{base64.b64encode(open("tea_grade_app/assests/home_page.jpg", "rb").read()).decode()}" 
            style="width: 100%; max-width: 1100px; max-height: 400px; height: auto; object-fit: cover; display: block;">
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="max-width: 1100px; margin: 2.5rem auto 3rem; padding: 2rem 2.5rem; background: linear-gradient(135deg, #d9f0e1, #a9d08e); border-radius: 20px; box-shadow: 0 8px 20px rgba(46, 125, 50, 0.2); font-family: 'Lato', sans-serif; color: #1b3a1b; text-align: center;">
            <h2 style="font-family: 'Forum', serif; font-weight: 700; font-size: 2.5rem; margin-bottom: 1rem; color: #134e13; text-shadow: 1px 1px 3px rgba(0,0,0,0.1);">
                How to Use <span style="color:#4caf50;">BrewGrade</span>
            </h2>
            <p style="font-size: 1.2rem; font-weight: 600; max-width: 720px; margin: 0 auto 2rem; line-height: 1.6; color: #2f552f;">
                Take a photo of your tea leaf and let our AI do the grading magic!<br>
                Whether you're a farmer or a tea producer, <strong>BrewGrade</strong> helps you instantly identify the grade of your tea from a single image.
            </p>
            <p style="font-size: 1.15rem; font-weight: 600; margin-bottom: 1rem; color: #3a723a;">
                Our AI classifies tea leaves based on the Pekoe Grading System, sorting them into:
            </p>
            <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap; margin-bottom: 2rem;">
                <div style="background: #a5d6a7; border-radius: 12px; box-shadow: 0 4px 10px rgba(69, 139, 36, 0.4); padding: 1rem 1.5rem; min-width: 110px; font-weight: 700; font-size: 1.3rem; color: #1b3a1b;">
                    FTGFOP
                </div>
                <div style="background: #81c784; border-radius: 12px; box-shadow: 0 4px 10px rgba(56, 142, 60, 0.4); padding: 1rem 1.5rem; min-width: 90px; font-weight: 700; font-size: 1.3rem; color: #1b3a1b;">
                    GFOP
                </div>
                <div style="background: #66bb6a; border-radius: 12px; box-shadow: 0 4px 10px rgba(46, 125, 50, 0.4); padding: 1rem 1.5rem; min-width: 70px; font-weight: 700; font-size: 1.3rem; color: #1b3a1b;">
                    OP
                </div>
                <div style="background: #ef9a9a; border-radius: 12px; box-shadow: 0 4px 10px rgba(211, 47, 47, 0.4); padding: 1rem 1.5rem; min-width: 90px; font-weight: 700; font-size: 1.3rem; color: #621010;">
                    Reject
                </div>
            </div>
            <div style="font-size: 1.1rem; line-height: 1.6; text-align: left; max-width: 500px; margin: 0 auto; color: #276627;">
                <ul style="list-style-type: none; padding-left: 0;">
                    <li style="margin-bottom: 0.8rem; display: flex; align-items: center;">
                        <span style="font-size: 1.5rem; margin-right: 0.6rem;"></span>
                        <strong>Step 1: </strong> Snap a clear picture of a tea leaf
                    </li>
                    <li style="margin-bottom: 0.8rem; display: flex; align-items: center;">
                        <span style="font-size: 1.5rem; margin-right: 0.6rem;"></span>
                        <strong>Step 2: </strong> Upload it under the Tea Leaf Recognition tab
                    </li>   
                    <li style="margin-bottom: 0.8rem; display: flex; align-items: center;">
                        <span style="font-size: 1.5rem; margin-right: 0.6rem;"></span>
                        <strong>Step 3: </strong> Click Predict to view the result
                    </li>
                </ul>
            </div>
            <p style="font-size: 1.1rem; font-weight: 600; margin-top: 2rem; color: #2f5b2f; max-width: 600px; margin-left: auto; margin-right: auto; line-height: 1.5;">
                Perfect for improving quality control and boosting confidence in every batch.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# === About Page ===
elif app_mode == "About":
    # Inject custom fonts and base styles
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Forum&display=swap');

    .section-title {
        background: linear-gradient(90deg, #439d59 0%, #b0eacb 60%);
        padding: 12px;
        border-radius: 12px;
        font-size: 1.7rem;
        font-family: 'Forum', serif;
        color: #fff;
        margin-bottom: 14px;
        text-align: center;
        box-shadow: 0 2px 8px #439d5944;
        letter-spacing: 0.5px;
    }
    .content-block {
        background: #e8f6ef;
        padding: 18px;
        border-radius: 14px;
        font-size: 1.09rem;
        margin-bottom: 28px;
        font-family: 'Forum', serif;
        line-height: 1.6;
        box-shadow: 0 2px 6px #b6dfb680;
        color: #1a3e2e;
    }
    .grade-card {
        background: linear-gradient(135deg, #b1e4c1 60%, #43a047 100%);
        border-radius: 16px;
        padding: 18px 12px 14px 12px;
        margin-bottom: 18px;
        box-shadow: 0 4px 12px #18923722;
        text-align: center;
        font-family: 'Forum', serif;
        color: #174b2d;
        transition: box-shadow 0.2s;
    }
    .grade-header {
        font-size: 1.21rem;
        font-weight: 700;
        color: #1a5836;
        margin-bottom: 0.2rem;
        letter-spacing: 0.5px;
    }
    .grade-title {
        color: #2e7253;
        font-size: 1.07rem;
        font-weight: 700;
        margin-bottom: 2px;
    }
    .grade-desc {
        font-size: 0.98rem;
        color: #22623b;
        font-weight: 400;
    }
    .reject-card {
        background: linear-gradient(135deg, #dce4da 60%, #b4b7aa 100%);
        color: #93473a;
        box-shadow: 0 4px 12px #d7747422;
    }
    .reject-header, .reject-title {
        color: #93473a;
    }
    .reject-desc {
        color: #b35540;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>🍃 About Pekoe Tea Grading 🍃</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='content-block'>
        Pekoe tea grading is an <b>internationally recognized system</b> primarily used in <b>Sri Lanka</b> and <b>India</b> to classify <b>black tea leaves</b> based on their size, appearance, and overall quality.<br><br>
        This grading helps <b>growers</b>, <b>producers</b>, and <b>buyers</b> instantly understand the value and intended use of tea batches at a glance!
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>The Main Pekoe Grades</div>", unsafe_allow_html=True)

    # 4 responsive columns for the grades
    grade_cols = st.columns(4)
    with grade_cols[0]:
        st.markdown("""
        <div class='grade-card'>
            <div class='grade-header'>🥇 FTGFOP</div>
            <div class='grade-title'>Finest Tippy Golden Flowery Orange Pekoe</div>
            <div class='grade-desc'>
                • Highest grade <br>
                • Long, twisted leaves <br>
                • Abundant golden tips <br>
                • Exceptional flavor
            </div>
        </div>
        """, unsafe_allow_html=True)
    with grade_cols[1]:
        st.markdown("""
        <div class='grade-card'>
            <div class='grade-header'>🥈 GFOP</div>
            <div class='grade-title'>Golden Flowery Orange Pekoe</div>
            <div class='grade-desc'>
                • High quality, some golden tips<br>
                • Great taste & aroma<br>
                • For premium blends
            </div>
        </div>
        """, unsafe_allow_html=True)
    with grade_cols[2]:
        st.markdown("""
        <div class='grade-card'>
            <div class='grade-header'>🟠 OP</div>
            <div class='grade-title'>Orange Pekoe</div>
            <div class='grade-desc'>
                • Long, wiry leaves<br>
                • No golden tips<br>
                • Standard black tea grade
            </div>
        </div>
        """, unsafe_allow_html=True)
    with grade_cols[3]:
        st.markdown("""
        <div class='grade-card reject-card'>
            <div class='grade-header reject-header'>❌ Reject</div>
            <div class='grade-title reject-title'>Reject</div>
            <div class='grade-desc reject-desc'>
                • Not meeting Pekoe standard<br>
                • Leaves, stems, or broken material<br>
                • Used in low-grade or dust teas
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>How Are Tea Grades Determined?</div>", unsafe_allow_html=True)

    with st.expander(" Click to view visual grading criteria"):
        st.markdown("""
        - **Shape & Size**: Whole, unbroken leaves = higher quality  
        - **Tips**: Golden or silver tips = young & delicate shoots  
        - **Uniformity**: Consistent leaf size and color = trusted grading  
        - **Color**: Vibrant hues & golden shine = freshness + skilled processing
        """)

    st.markdown("<div class='section-title'>Why It Matters</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='content-block'>
        ✅ Helps <b>farmers</b> and <b>producers</b> get fair value for high-quality harvests<br>
        ✅ Assists <b>buyers</b> in selecting the right tea for their needs<br>
        ✅ Builds <b>trust and transparency</b> in the global tea trade
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='section-title'> TeaVision brings this expert system to your fingertips</div>
    <div class='content-block' style='text-align:center;'>
        Helping you understand, grade, and celebrate your tea, one leaf at a time!
    </div>
    """, unsafe_allow_html=True)

# === Tea Leaf Recognition Page ===
elif app_mode == "Tea Grading":
    # Import font once (you can move this to app start or About page styles if preferred)
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Forum&display=swap');

        .title-forum {
            font-family: 'Forum', serif !important;
            font-weight: 700;
            font-size: 2.6rem;
            color: #236b31;  /* matching your green theme */
            margin-bottom: 0.4rem;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    # Use this CSS class for the title text
    st.markdown('<div class="title-forum"> Tea Leaf Recognition</div>', unsafe_allow_html=True)

    st.info("Upload a high-quality tea leaf image for best results.")

    uploaded_file = st.file_uploader("Upload your tea leaf image", type=["jpg", "jpeg", "png"], key="file_uploader")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([3, 2])

        with col1:
            # You can style the image container similarly as desired
            st.image(image, caption="🖼️ Uploaded Tea Leaf Image", use_container_width=True)

        with col2:
            if st.button("🔍 Predict", key="predict_button"):
                with st.spinner("⏳ Classifying..."):
                    prediction = predict_image(image)

                st.success(" Prediction Complete!")
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, #b1e4c1 60%, #43a047 100%);
                        border-radius: 15px;
                        padding: 2rem 1.2rem;
                        box-shadow: 0 5px 15px rgba(24, 146, 55, 0.25);
                        text-align: center;
                        font-family: 'Forum', serif;
                        color: #174b2d;
                        max-width: 280px;
                        margin: 0 auto;
                    ">
                        <h3 style='font-weight: 700; margin-bottom: 0.5rem;'>Predicted Grade</h3>
                        <h1 style='font-size: 3rem; margin: 0; color: #0f3d18;'>{prediction}</h1>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
