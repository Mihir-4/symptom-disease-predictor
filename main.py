import streamlit as st
import joblib
from streamlit_image_coordinates import streamlit_image_coordinates

from app.utils import (
    load_model,
    load_encoder,
    predict_top3
)
from app.body_regions import (
    BODY_REGIONS
)
from app.load_symptoms import (
    ALL_SYMPTOMS
)

st.set_page_config(
    page_title="Symptom-to-Disease Predictor",
    page_icon="🩺",
    layout="wide"
)

# --------------------------------------------------
# CSS THEME CUSTOM INTEGRATION
# --------------------------------------------------
st.markdown("""
<style>
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        padding: 10px 0;
    }
    .symptom-badge {
        display: inline-flex;
        align-items: center;
        background-color: #1e222b;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 4px;
        padding: 6px 12px;
        font-size: 14px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD RESOURCES
# --------------------------------------------------
@st.cache_resource
def load_resources():
    model = load_model()
    encoder = load_encoder()
    feature_columns = joblib.load("models/feature_columns.pkl")
    return model, encoder, feature_columns

model, encoder, feature_columns = load_resources()

# --------------------------------------------------
# SESSION STATE INITIALIZATION
# --------------------------------------------------
if "selected_symptoms" not in st.session_state:
    st.session_state.selected_symptoms = set()

if "active_region" not in st.session_state:
    st.session_state.active_region = None

# --------------------------------------------------
# HEADER & METRICS
# --------------------------------------------------
st.title("🩺 Symptom-to-Disease Predictor")
st.caption("AI Powered Disease Prediction using XGBoost")

c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Diseases", "677")
with c2: st.metric("Symptoms", "328")
with c3: st.metric("Accuracy", "83.51%")
with c4: st.metric("Top-3 Accuracy", "94.55%")

st.divider()

# --------------------------------------------------
# BODY + SEARCH
# --------------------------------------------------
left, right = st.columns([1, 1])

with left:
    st.markdown("### 🧍 Body Region Selection")
    
    if st.session_state.active_region:
        st.markdown(f"#### Active Region: `{st.session_state.active_region}`")
        if st.button("Clear Filter (Show All)", use_container_width=True):
            st.session_state.active_region = None
            st.rerun()
    else:
        st.markdown("<p style='color: #666; font-size: 15px; margin-bottom: 24px;'>Showing all symptom profiles. Click a zone to refine.</p>", unsafe_allow_html=True)
    
    value = streamlit_image_coordinates(
        "assets/body_silhouette.png",
        width=500,
        key="body_click_canvas"
    )

    if value:
        x = value["x"]
        y = value["y"]
        region = "Unknown"

        if 180 <= x <= 320 and 0 <= y <= 155:
            region = "Head"
        elif 160 <= x <= 340 and 156 <= y <= 295:
            region = "Chest"
        elif 160 <= x <= 340 and 296 <= y <= 440:
            region = "Abdomen"
        elif (50 <= x <= 160 and 160 <= y <= 410) or (340 <= x <= 450 and 160 <= y <= 410):
            region = "Arms"
        elif 150 <= x <= 350 and y >= 441:
            region = "Legs"

        if region != "Unknown" and st.session_state.active_region != region:
            st.session_state.active_region = region
            st.rerun()

with right:
    st.markdown("### 🔍 Search Symptoms")

    if st.session_state.active_region:
        raw_source = BODY_REGIONS.get(st.session_state.active_region, [])
        placeholder_text = f"Type to look up {st.session_state.active_region} symptoms..."
    else:
        raw_source = ALL_SYMPTOMS
        placeholder_text = "Select an anatomy zone or look up symptoms here..."

    # FILTER FIX: Subtract already selected symptoms so they don't appear in the dropdown
    search_source = [symptom for symptom in raw_source if symptom not in st.session_state.selected_symptoms]

    def handle_symptom_selection():
        if st.session_state.symptom_search_input:
            for item in st.session_state.symptom_search_input:
                st.session_state.selected_symptoms.add(item)
            st.session_state.symptom_search_input = []

    st.multiselect(
        label=placeholder_text,
        options=search_source,
        key="symptom_search_input",
        on_change=handle_symptom_selection,
        label_visibility="collapsed"
    )

# --------------------------------------------------
# SELECTED SYMPTOMS BADGES
# --------------------------------------------------
st.divider()
st.markdown("### ✅ Selected Symptoms")

if st.session_state.selected_symptoms:
    badge_html = "<div class='badge-container'>"
    for symptom in st.session_state.selected_symptoms:
        badge_html += f"<div class='symptom-badge'>{symptom}</div>"
    badge_html += "</div>"
    st.markdown(badge_html, unsafe_allow_html=True)
    
    cols = st.columns(min(len(st.session_state.selected_symptoms), 4))
    for idx, symptom in enumerate(list(st.session_state.selected_symptoms)):
        col_target = cols[idx % 4]
        with col_target:
            if st.button(f"Remove: {symptom}", key=f"rm_{symptom}", use_container_width=True):
                st.session_state.selected_symptoms.remove(symptom)
                st.rerun()
else:
    st.markdown("<p style='color: #666; font-style: italic;'>No symptoms selected yet.</p>", unsafe_allow_html=True)

st.metric("Total Selected Variables", len(st.session_state.selected_symptoms))

# --------------------------------------------------
# DIAGNOSTICS & PREDICTIONS
# --------------------------------------------------
if st.button("Predict Disease", use_container_width=True):
    if len(st.session_state.selected_symptoms) == 0:
        st.markdown("<p style='color: #888;'>Please add symptoms before executing prediction routines.</p>", unsafe_allow_html=True)
    else:
        results = predict_top3(
            model,
            encoder,
            list(st.session_state.selected_symptoms),
            feature_columns
        )

        best = results[0]
        st.divider()

        with st.container():
            st.markdown(f"""
            ### 🎯 Most Likely Diagnosis
            ## {best['disease'].title()}
            * **Model Probability:** {best['probability'] * 100:.2f}%
            * **Confidence Metric:** {best['confidence']}
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Top 3 Alternative Forecasts")
        col1, col2, col3 = st.columns(3)
        cards = [col1, col2, col3]

        for i, result in enumerate(results):
            with cards[i]:
                st.metric(
                    label=result["disease"].title(),
                    value=f"{result['probability'] * 100:.2f}%"
                )
                st.markdown(f"<small style='color:#888;'>Confidence: {result['confidence']}</small>", unsafe_allow_html=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption("Educational use only. Not a substitute for professional medical advice.")