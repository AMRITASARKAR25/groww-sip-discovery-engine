import os
import json
import time
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

st.set_page_config(
    page_title="SIP Investor Discovery Engine",
    page_icon="📊",
    layout="wide"
)

st.title("📊 SIP Investor Discovery Engine")
st.caption("AI-powered research | Groww SIP continuation")

RESULTS_FILE = Path("sip_ai_analysis.csv")
MODEL = "gemini-3.1-flash-lite"
BATCH_SIZE = 5

FIELDS = [
    "investor_stage",
    "motivation",
    "sip_behavior",
    "main_concern",
    "theme",
    "workaround",
    "evidence_confidence"
]

ALLOWED = {
    "investor_stage": {
        "first_time_investor", "experienced", "unknown"
    },
    "sip_behavior": {
        "continuing", "paused", "stopped",
        "considering_stopping", "unknown"
    },
    "theme": {
        "return_expectations",
        "financial_constraints",
        "payment_failure",
        "investment_confusion",
        "app_experience",
        "external_influence",
        "positive_experience",
        "other"
    },
    "evidence_confidence": {
        "high", "medium", "low"
    }
}

try:
    api_key = st.secrets.get("GEMINI_API_KEY")
except (FileNotFoundError, KeyError):
    api_key = None
api_key = api_key or os.getenv("GEMINI_API_KEY")

client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(timeout=90000)
) if api_key else None


def load_results():
    if RESULTS_FILE.exists():
        return pd.read_csv(
            RESULTS_FILE,
            dtype={"feedback_id": str}
        ).fillna("unknown")
    return pd.DataFrame()


def save_results(dataframe):
    temporary = RESULTS_FILE.with_suffix(".tmp")
    dataframe.to_csv(temporary, index=False)
    os.replace(temporary, RESULTS_FILE)


def is_daily_quota_error(error):
    message = str(error).lower()

    return (
        "429" in message
        and (
            "per day" in message
            or "requests per day" in message
            or "generaterequestsperday" in message
            or "free_tier_requests" in message
        )
    )


def analyze_batch(batch_df):
    if client is None:
        raise RuntimeError("Live AI analysis is not configured. The reference research remains available.")

    records = [
        {
            "feedback_id": str(row["feedback_id"]),
            "feedback_text": str(row["feedback_text"])
        }
        for _, row in batch_df.iterrows()
    ]

    prompt = """
You are a product research analyst examining SIP continuation
among mutual fund investors in India.

Classify EACH feedback record independently.

Return a JSON array containing exactly one object per input record.

Each object must contain:
feedback_id
investor_stage
motivation
sip_behavior
main_concern
theme
workaround
evidence_confidence

Allowed investor_stage:
first_time_investor, experienced, unknown

Allowed sip_behavior:
continuing, paused, stopped, considering_stopping, unknown

Allowed theme:
return_expectations, financial_constraints, payment_failure,
investment_confusion, app_experience, external_influence,
positive_experience, other

Allowed evidence_confidence:
high, medium, low

Rules:
- Preserve each feedback_id exactly.
- Classify each feedback independently using only its feedback_text.
- Do not invent facts, motivations, emotions, or investment behavior.

INVESTOR STAGE:
- Use first_time_investor only when the person explicitly says
  this is their first mutual fund investment or first SIP.
- Use experienced only when there is explicit evidence of
  previous investing experience.
- Otherwise use unknown.
- Being a beginner does not automatically mean first-time investor.

SIP BEHAVIOR:
- Use continuing only when the feedback explicitly confirms
  repeated or ongoing SIP instalments.
- Use paused only when the investor explicitly describes
  temporarily pausing their SIP.
- Use stopped only when the investor explicitly confirms
  stopping or cancelling their SIP.
- Use considering_stopping only when the investor explicitly
  expresses an intention to stop.
- Starting a SIP, planning future investments, or asking
  questions about SIPs does not prove continuation.
- Otherwise use unknown.

MOTIVATION AND CONCERN:
- Identify motivations and concerns only when supported
  by the feedback.
- Do not assume poor returns caused dissatisfaction.
- If the motivation or concern is not stated, use unknown.

EVIDENCE:
- General advice is not necessarily personal experience.
- Use high confidence only when the classification is
  directly supported by explicit statements.
- Use medium when evidence is indirect but reasonably clear.
- Use low when evidence is limited or ambiguous.
- Never treat an unverified paraphrase as a verified quotation.

- Do not recommend funds or stocks.
- Return valid JSON only.
INPUT RECORDS:
""" + json.dumps(records, ensure_ascii=False)

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0
        )
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    predictions = json.loads(response.text)

    if not isinstance(predictions, list):
        raise ValueError("Expected a JSON array.")

    expected_ids = set(batch_df["feedback_id"].astype(str))
    received_ids = []

    for prediction in predictions:

        if not isinstance(prediction, dict):
            raise ValueError("Invalid classification object.")

        feedback_id = str(prediction.get("feedback_id", ""))
        received_ids.append(feedback_id)

        for field in FIELDS:
            if field not in prediction:
                raise ValueError(
                    f"Missing {field} for {feedback_id}"
                )

            if not isinstance(prediction[field], str):
                raise ValueError(
                    f"Invalid value type for {field}"
                )

        for field, allowed_values in ALLOWED.items():
            if prediction[field] not in allowed_values:
                raise ValueError(
                    f"Invalid {field} for {feedback_id}"
                )

    if (
        len(received_ids) != len(expected_ids)
        or len(received_ids) != len(set(received_ids))
        or set(received_ids) != expected_ids
    ):
        raise ValueError(
            "Gemini returned missing, duplicate, or unexpected IDs."
        )

    return predictions


def analyze_batch_with_retry(batch_df):

    for attempt in range(2):

        try:
            return analyze_batch(batch_df)

        except Exception as error:

            if is_daily_quota_error(error):
                raise RuntimeError(
                    "DAILY_QUOTA_EXHAUSTED: "
                    "Stop analysis until your quota resets."
                ) from error

            message = str(error)

            temporary_error = any(
                code in message
                for code in ["503", "500", "502", "504"]
            )

            if not temporary_error or attempt == 1:
                raise

            st.warning(
                "Temporary Gemini server error. "
                "Retrying once in 10 seconds."
            )

            time.sleep(10)



REFERENCE_FILE = Path(__file__).parent / "reference_analysis.csv"

st.header("1. Explore the research dataset")
st.caption("Pre-analyzed research reference dataset · 26 feedback records · qualitative pilot")
if REFERENCE_FILE.exists():
    reference_df = pd.read_csv(REFERENCE_FILE, dtype={"feedback_id": str}).fillna("")
    c1, c2, c3 = st.columns(3)
    c1.metric("Analyzed feedback records", len(reference_df))
    c2.metric("Research themes", reference_df["theme"].nunique())
    c3.metric("Feedback sources", reference_df["source"].nunique())
    st.subheader("Theme distribution")
    st.bar_chart(reference_df["theme"].value_counts())
    st.subheader("SIP behavior distribution")
    st.bar_chart(reference_df["sip_behavior"].value_counts())
    st.subheader("Explore individual evidence")
    themes = sorted(reference_df["theme"].dropna().unique())
    chosen_themes = st.multiselect("Filter by AI theme", themes, default=themes)
    filtered = reference_df[reference_df["theme"].isin(chosen_themes)]
    query = st.text_input("Search feedback text / keywords", "")
    if query:
        filtered = filtered[filtered["feedback_text"].astype(str).str.contains(query, case=False, regex=False, na=False)]
    display_cols = [c for c in ["feedback_id", "source", "feedback_text", "investor_stage", "sip_behavior", "theme", "main_concern", "source_url", "mutual_fund_sip_relevance"] if c in filtered.columns]
    st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)
    st.download_button("Download reference classifications", reference_df.to_csv(index=False).encode("utf-8"), "reference_analysis.csv", "text/csv")
    st.info("These are exploratory qualitative records, including paraphrases and adjacent investing discussions. They do not measure Groww's actual 12-month SIP retention rate.")
else:
    st.warning("Reference dataset not found. Add reference_analysis.csv beside app.py in the GitHub repository.")

st.divider()
st.header("2. Test the AI Discovery Engine")
st.write("Upload feedback in CSV format to run new Gemini classifications. Each visitor's test is isolated to their current session.")
if not api_key:
    st.warning("Live analysis is unavailable because the Gemini API key is not configured in Streamlit Secrets. The reference dashboard remains accessible.")

st.download_button("Download sample input CSV", "feedback_id,feedback_text\nTEST-001,This is my first mutual fund SIP and I am confused about the automatic debit.\nTEST-002,I have paid monthly SIP instalments for two years.\n", "sample_feedback.csv", "text/csv")
uploaded_file = st.file_uploader("Upload CSV containing feedback_id and feedback_text", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, dtype={"feedback_id": str})
    except Exception as error:
        st.error(f"Cannot read CSV: {error}")
        st.stop()
    if not {"feedback_id", "feedback_text"}.issubset(df.columns):
        st.error("CSV must contain feedback_id and feedback_text columns.")
        st.stop()
    if df.empty or df["feedback_id"].isna().any() or df["feedback_id"].duplicated().any() or df["feedback_text"].isna().any():
        st.error("CSV must have nonempty, unique feedback IDs and nonempty feedback text.")
        st.stop()
    import hashlib
    fingerprint = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    if st.session_state.get("upload_fingerprint") != fingerprint:
        st.session_state["upload_fingerprint"] = fingerprint
        st.session_state["visitor_results"] = pd.DataFrame()
    saved_df = st.session_state["visitor_results"]
    completed_ids = set(saved_df["feedback_id"].astype(str)) if not saved_df.empty else set()
    remaining_df = df[~df["feedback_id"].isin(completed_ids)]
    x, y, z = st.columns(3)
    x.metric("Uploaded records", len(df))
    y.metric("Analyzed in this session", len(completed_ids))
    z.metric("Remaining", len(remaining_df))
    st.progress(min(len(completed_ids) / len(df), 1.0))
    if len(remaining_df):
        st.caption(f"Up to {BATCH_SIZE} records per Gemini request. Temporary server errors do not erase earlier successful batches.")
        if st.button(f"Analyze next {min(BATCH_SIZE, len(remaining_df))} records", disabled=not bool(api_key)):
            batch_df = remaining_df.head(BATCH_SIZE)
            with st.spinner("Classifying feedback with Gemini..."):
                try:
                    predictions = analyze_batch_with_retry(batch_df)
                    new_results = batch_df.merge(pd.DataFrame(predictions), on="feedback_id", validate="one_to_one")
                    st.session_state["visitor_results"] = pd.concat([saved_df, new_results], ignore_index=True)
                    st.rerun()
                except Exception as error:
                    st.error(f"Analysis unavailable: {error}")
                    st.info("Previously completed batches remain in this browser session. Try again later if Gemini is overloaded.")
    else:
        st.success("All uploaded records have been analyzed in this session.")
    if not saved_df.empty:
        st.subheader("Your AI classification results")
        st.dataframe(saved_df, use_container_width=True, hide_index=True)
        st.download_button("Download your analyzed CSV", saved_df.to_csv(index=False).encode("utf-8"), "sip_ai_test_results.csv", "text/csv")

st.divider()
st.caption("Research purpose: understand potential SIP continuation barriers. No specific investment recommendations are generated.")
