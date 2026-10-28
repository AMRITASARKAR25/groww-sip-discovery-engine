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

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("Gemini API key missing from .env")
    st.stop()

client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(timeout=60000)
)


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


st.header("1. Upload investor feedback")

uploaded_file = st.file_uploader(
    "Upload your original 26-record CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(
        uploaded_file,
        dtype={"feedback_id": str}
    )

    if not {"feedback_id", "feedback_text"}.issubset(df.columns):
        st.error(
            "CSV needs feedback_id and feedback_text columns."
        )
        st.stop()

    if (
        df["feedback_id"].isna().any()
        or df["feedback_id"].duplicated().any()
    ):
        st.error("Feedback IDs must be unique and nonempty.")
        st.stop()

    st.success("Dataset uploaded successfully.")
    st.dataframe(df, use_container_width=True)

    saved_df = load_results()

    if not saved_df.empty:
        if (
            "feedback_id" not in saved_df.columns
            or saved_df["feedback_id"].duplicated().any()
        ):
            st.error("Saved results contain invalid feedback IDs.")
            st.stop()

        saved_df = saved_df[
            saved_df["feedback_id"].isin(df["feedback_id"])
        ].copy()

    completed_ids = (
        set(saved_df["feedback_id"])
        if not saved_df.empty
        else set()
    )

    remaining_df = df[
        ~df["feedback_id"].isin(completed_ids)
    ].copy()

    st.header("2. AI feedback analysis")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total records", len(df))
    col2.metric("Already analyzed", len(completed_ids))
    col3.metric("Remaining", len(remaining_df))

    st.progress(len(completed_ids) / len(df))

    if len(remaining_df) > 0:

        st.info(
            "Each request analyzes up to 5 remaining records. "
            "Completed records will not be sent again."
        )

        if st.button("Analyze next batch of 5"):

            batch_df = remaining_df.head(BATCH_SIZE)

            with st.spinner(
                f"Analyzing {len(batch_df)} records..."
            ):

                try:
                    predictions = analyze_batch_with_retry(
                        batch_df
                    )

                    prediction_df = pd.DataFrame(predictions)

                    # Preserve original evidence and source fields.
                    source_df = batch_df.copy()

                    new_results = source_df.merge(
                        prediction_df,
                        on="feedback_id",
                        how="left",
                        validate="one_to_one"
                    )

                    saved_df = pd.concat(
                        [saved_df, new_results],
                        ignore_index=True
                    )

                    saved_df = saved_df.drop_duplicates(
                        subset=["feedback_id"],
                        keep="last"
                    )

                    save_results(saved_df)

                    st.success(
                        f"Saved {len(new_results)} new records!"
                    )

                    st.info(
                        "Click Refresh progress to update the counts."
                    )

                except Exception as error:

                    if "DAILY_QUOTA_EXHAUSTED" in str(error):
                        st.error(
                            "Daily Gemini quota exhausted. "
                            "Stop here and try after the reset."
                        )
                    else:
                        st.error(
                            f"Batch failed: {error}"
                        )

                    st.warning(
                        "No records from this failed batch were saved. "
                        "Previous results are safe."
                    )

            st.button(
                "Refresh progress",
                on_click=st.rerun
            )

    else:
        st.success("All records have been analyzed!")

    st.header("3. Saved AI classification results")

    results_df = load_results()

    if not results_df.empty:

        results_df = results_df[
            results_df["feedback_id"].isin(df["feedback_id"])
        ]

        st.dataframe(
            results_df,
            use_container_width=True
        )

        if "theme" in results_df.columns:

            st.subheader("Theme distribution")

            st.bar_chart(
                results_df["theme"].value_counts()
            )

        st.download_button(
            "Download analyzed CSV",
            data=results_df.to_csv(index=False),
            file_name="sip_ai_analysis_complete.csv",
            mime="text/csv"
        )