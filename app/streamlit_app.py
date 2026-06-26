from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_ID = "Joshua2565/complaint-radar-distilbert"
LOCAL_MODEL_DIR = Path(__file__).resolve().parents[1] / "complaint-radar-distilbert"

ID_TO_GROUP = {
    0: "Banking",
    1: "Credit card / prepaid card",
    2: "Credit reporting",
    3: "Debt collection",
    4: "Debt or credit management",
    5: "Money transfer / virtual currency",
    6: "Mortgage",
    7: "Other financial service",
    8: "Payday / personal / consumer loan",
    9: "Student loan",
    10: "Vehicle loan / lease",
}

LIKELY_TEXT_COLUMNS = [
    "text",
    "complaint",
    "complaint_text",
    "consumer_complaint",
    "consumer_complaint_narrative",
    "narrative",
    "description",
    "issue",
]


st.set_page_config(
    page_title="Complaint Radar",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading Complaint Radar model...")
def load_model():
    model_source = LOCAL_MODEL_DIR if LOCAL_MODEL_DIR.exists() else MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForSequenceClassification.from_pretrained(model_source)
    model.eval()
    return tokenizer, model, str(model_source)


def find_default_text_column(columns: list[str]) -> str | None:
    normalized = {column.strip().lower(): column for column in columns}
    for candidate in LIKELY_TEXT_COLUMNS:
        if candidate in normalized:
            return normalized[candidate]
    return None


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    try:
        return pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="latin-1")


def predict_groups(
    texts: list[str], tokenizer,model,batch_size: int = 16,) -> tuple[list[int], list[float]]:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    predicted_ids: list[int] = []
    confidence_scores: list[float] = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            confidence, predictions = torch.max(probabilities, dim=-1)

            predicted_ids.extend(predictions.cpu().tolist())
            confidence_scores.extend(confidence.cpu().tolist())

    return predicted_ids, confidence_scores


def build_download(results_df: pd.DataFrame) -> bytes:
    buffer = StringIO()
    results_df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
    .radar-title {
        font-size: 2.4rem;
        font-weight: 750;
        margin-bottom: 0.25rem;
    }
    .radar-subtitle {
        color: #46515f;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
        max-width: 860px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="radar-title">Complaint Radar</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="radar-subtitle">
    Upload a CSV of finance-based customer complaints and classify each complaint into
    its predicted financial service group.
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload finance customer complaints CSV",
    type=["csv"],
    help="The dashboard expects one column containing complaint narrative text.",
)

if not uploaded_file:
    example = pd.DataFrame(
        {
            "text": [
                "My credit report still shows an account I already disputed.",
                "The mortgage servicer charged fees I do not recognize.",
                "A debt collector keeps calling about a debt that is not mine.",
            ]
        }
    )
    st.info("Upload a CSV to start. A column named `text` will be detected automatically.")
    st.dataframe(example, use_container_width=True, hide_index=True)
    st.stop()

try:
    complaints_df = read_uploaded_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the uploaded CSV: {exc}")
    st.stop()

if complaints_df.empty:
    st.warning("The uploaded CSV is empty.")
    st.stop()

default_text_column = find_default_text_column(list(complaints_df.columns))
column_options = list(complaints_df.columns)
default_index = column_options.index(default_text_column) if default_text_column else 0

with st.sidebar:
    st.header("Prediction Settings")
    text_column = st.selectbox(
        "Complaint text column",
        options=column_options,
        index=default_index,
    )
    batch_size = st.slider("Batch size", min_value=4, max_value=64, value=16, step=4)
    run_predictions = st.button("Predict Complaint Groups", type="primary", use_container_width=True)

    st.divider()
    st.caption("Model")
    st.code(MODEL_ID)

valid_rows = complaints_df[text_column].notna() & complaints_df[text_column].astype(str).str.strip().ne("")
complaint_count = int(valid_rows.sum())

summary_left, summary_mid, summary_right = st.columns(3)
summary_left.metric("Rows uploaded", f"{len(complaints_df):,}")
summary_mid.metric("Complaints ready", f"{complaint_count:,}")
summary_right.metric("Finance groups", len(ID_TO_GROUP))

st.subheader("Uploaded Complaints")
st.dataframe(complaints_df.head(25), use_container_width=True, hide_index=True)

if not run_predictions:
    st.stop()

if complaint_count == 0:
    st.warning("No usable complaint text was found in the selected column.")
    st.stop()

try:
    tokenizer, model, model_source = load_model()
except Exception as exc:
    st.error(
        "The model could not be loaded. Make sure the local model folder exists or "
        "that this machine can reach Hugging Face."
    )
    st.exception(exc)
    st.stop()

results_df = complaints_df.copy()
texts = results_df.loc[valid_rows, text_column].astype(str).tolist()

with st.spinner("Classifying finance complaints..."):
    predicted_ids, confidence_scores = predict_groups(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        batch_size=batch_size,
    )

results_df["predicted_group_id"] = pd.NA
results_df["predicted_group_name"] = pd.NA
results_df["prediction_confidence"] = pd.NA
results_df.loc[valid_rows, "predicted_group_id"] = predicted_ids
results_df.loc[valid_rows, "predicted_group_name"] = [
    ID_TO_GROUP.get(group_id, "Unknown financial service") for group_id in predicted_ids
]
results_df.loc[valid_rows, "prediction_confidence"] = [
    round(score, 4) for score in confidence_scores
]

category_counts = (
    results_df.loc[valid_rows, "predicted_group_name"]
    .value_counts()
    .rename_axis("Financial complaint group")
    .reset_index(name="Complaint count")
)
category_counts["Percent of complaints"] = (
    category_counts["Complaint count"] / category_counts["Complaint count"].sum() * 100
).round(2)

st.success(f"Classified {complaint_count:,} finance complaints using `{model_source}`.")

chart_col, table_col = st.columns([1.3, 1])
with chart_col:
    st.subheader("Category Percent")
    fig = px.bar(
        category_counts.sort_values("Percent of complaints"),
        x="Percent of complaints",
        y="Financial complaint group",
        orientation="h",
        text="Percent of complaints",
        color="Financial complaint group",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        showlegend=False,
        xaxis_title="Percent of uploaded complaints",
        yaxis_title="",
        margin=dict(l=0, r=20, t=20, b=0),
        height=max(420, 34 * len(category_counts)),
    )
    st.plotly_chart(fig, use_container_width=True)

with table_col:
    st.subheader("Group Breakdown")
    st.dataframe(
        category_counts,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Percent of complaints": st.column_config.ProgressColumn(
                "Percent of complaints",
                format="%.2f%%",
                min_value=0,
                max_value=100,
            )
        },
    )

st.subheader("Predicted Complaint File")
st.dataframe(results_df.head(100), use_container_width=True, hide_index=True)

st.download_button(
    "Download CSV With Predicted Groups",
    data=build_download(results_df),
    file_name="finance_complaints_with_predicted_groups.csv",
    mime="text/csv",
    type="primary",
)
