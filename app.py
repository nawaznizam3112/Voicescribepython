import os

import streamlit as st
from google import genai
from google.genai import types


st.set_page_config(page_title="Voice Scribe", page_icon="🎙️")
st.title("🎙️ Clinical Voice Scribe")
st.caption("Drafts a chief complaint and HPI from a recorded encounter.")

st.warning(
    "Draft documentation only—not a diagnosis or substitute for clinical judgment. "
    "Review and correct all generated content before adding it to a medical record. "
    "Do not enter patient-identifying information unless this deployment is approved "
    "for that use."
)

def get_api_key():
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return None


with st.expander("Before recording", expanded=True):
    consent = st.checkbox(
        "I have obtained any required consent and am authorized to process this audio."
    )

language = st.text_input(
    "Encounter language",
    value="English",
    help="For example: English, Spanish, or Hindi.",
)

audio = st.audio_input("Record the patient encounter")

if st.button("Generate draft", type="primary", disabled=not consent):
    if audio is None:
        st.error("Record an encounter before generating a draft.")
        st.stop()

    audio_bytes = audio.getvalue()
    if not audio_bytes:
        st.error("The recording appears to be empty. Please record it again.")
        st.stop()

    # Keep inline audio within a conservative size limit.
    if len(audio_bytes) > 20 * 1024 * 1024:
        st.error("The recording is too large. Please record a shorter encounter.")
        st.stop()

    api_key = get_api_key()
    if not api_key:
        st.error(
            "Gemini API key not found. Set GEMINI_API_KEY as an environment "
            "variable or in Streamlit secrets."
        )
        st.stop()

    prompt = f"""
You are a clinical documentation assistant. The audio is an encounter recording
in {language}.

Create a draft note using only information clearly stated in the recording.
Do not invent, infer, or silently correct clinical facts. Preserve uncertainty.
Do not provide a diagnosis, differential diagnosis, treatment plan, or medical
advice. If speech is unclear, mark it as [unclear] rather than guessing.
Do not add details merely because they are commonly associated with a complaint.

Return the following sections in Markdown:

## Chief Complaint (CC)
A concise statement of the patient's stated reason for the visit.

## History of Present Illness (HPI)
A clear, chronological narrative of the complaint. Include relevant details
actually stated, such as onset, location, duration, character, severity,
context, aggravating or relieving factors, associated symptoms, and pertinent
positives or negatives. Omit details that were not provided.

## Relevant Positives and Negatives
List only symptoms explicitly mentioned in the recording. Write “Not stated”
when none are clear.

## Items to Verify
List unclear, contradictory, or potentially important details that a clinician
should confirm. Write “None identified” if applicable.

## Transcript
Provide a readable transcript. Mark unintelligible speech as [unclear].
"""

    try:
        client = genai.Client(api_key=api_key)
        with st.spinner("Generating a draft…"):
            response = client.models.generate_content(
                model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                contents=[
                    prompt,
                    types.Part.from_bytes(
                        data=audio_bytes,
                        mime_type=audio.type or "audio/wav",
                    ),
                ],
            )

        if not response.text:
            st.error("Gemini returned an empty response. Please try again.")
        else:
            st.subheader("Draft note — clinician review required")
            st.markdown(response.text)
            st.download_button(
                "Download draft as Markdown",
                data=response.text,
                file_name="clinical_note_draft.md",
                mime="text/markdown",
            )

    except Exception as exc:
        # Avoid displaying configuration details or request data.
        st.error(f"Could not generate the draft ({type(exc).__name__}). Please try again.")
