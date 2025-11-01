import streamlit as st
from PIL import Image
import tempfile
import cv2
import numpy as np
import pandas as pd
import altair as alt
from pathlib import Path
import io
import datetime

# --- FIX 1: Import type hints ---
from typing import Optional, List 
# --- FIX 3: Import for asynchronous handling ---
import asyncio 

# --- Telegram Imports ---
import requests
import telegram 
# --- FIX 2: Import constants for ParseMode ---
try:
    from telegram import constants
except ImportError:
    # Fallback to prevent crash if constants import fails
    class DummyConstants:
        class ParseMode:
            HTML = 'HTML'
    constants = DummyConstants()

# You might need to install 'python-telegram-bot' separately
try:
    # Dummy module setup to prevent crash if telegram is missing
    if not hasattr(telegram, 'Bot'):
        class DummyBot:
            def __init__(self, *args, **kwargs): pass
            def send_photo(self, *args, **kwargs): raise RuntimeError("Telegram not installed.")
        telegram.Bot = DummyBot
except Exception:
    pass


# --- PPE detection imports (your existing utils/detect.py) ---
try:
    # NOTE: You will need to modify these functions in utils/detect.py
    from utils.detect import detect_ppe_image, detect_ppe_video, load_model, get_model_labels
except ImportError:
    st.error("Fatal Error: `utils/detect.py` not found. Please ensure it's in the same directory.")
    st.stop()

# --- Face recognition & DB imports ---
FACE_RECOGNITION_AVAILABLE = True
try:
    import face_recognition
except Exception as e:
    FACE_RECOGNITION_AVAILABLE = False
    _face_error = e

# utils/face_db provides register_employee, load_all_encodings, log_violation, get_violation_count, get_recent_violations
try:
    from utils.face_db import (
        register_employee,
        load_all_encodings,
        log_violation,
        get_violation_count,
        get_recent_violations
    )
except ImportError:
    st.error("Fatal Error: `utils/face_db.py` not found. Please create it (see instructions).")
    st.stop()


EXAMPLE_IMAGE_PATH = "example_ppe.jpg"

st.set_page_config(
    page_title="PPE Detector",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🦺"
)

st.title("🦺 PPE Detection Pro")
st.markdown("AI-Powered Personal Protective Equipment Monitoring System")

with st.expander("👋 Welcome! Meet the Development Team", expanded=False):
    st.markdown("""
    **Welcome to PPE Detector Pro!** This app uses AI to monitor workplace safety.

    Developed by students from **IIT (BHU) Varanasi**:
    - **Ambuj Nayak** ([GitHub](https://github.com/Ambuj-N))
    - **Paturi Hemanth Sai**
    - **Ankit Raj**
    - **Jalla Poojitha**
    """)


# ---------------- TELEGRAM NOTIFICATION FUNCTION ----------------
BOT_TOKEN = "8258711886:AAGUwUmWsyrfHWpAWnhgieEL9ESobk_NsAs"
CHAT_ID = "1269174608"

# V V V FIX 3: ADD 'async' V V V
async def send_violation_notification(emp_id: Optional[str], name: Optional[str], missing_items: List[str], img_bytes: bytes):
    """Sends a violation alert to a Telegram chat."""
    
    if not hasattr(telegram, 'Bot') or not hasattr(constants.ParseMode, 'HTML'):
        st.toast("Telegram library not properly initialized/installed.", icon="❌")
        return

    name_str = name or "Unknown"
    id_str = emp_id or "N/A"
    missing_str = ", ".join(missing_items)
    
    # Format the message (supports HTML)
    message = (
        f"<b>🚨 PPE VIOLATION DETECTED 🚨</b>\n\n"
        f"<b>Name:</b> {name_str}\n"
        f"<b>ID:</b> {id_str}\n"
        f"<b>Missing:</b> {missing_str}"
    )
    
    try:
        bot = telegram.Bot(token=BOT_TOKEN)
        
        # Prepare image bytes for sending
        image_file = io.BytesIO(img_bytes)
        image_file.name = "violation.png"
        image_file.seek(0)
        
        # V V V FIX 3: ADD 'await' V V V
        await bot.send_photo(
            chat_id=CHAT_ID,
            photo=image_file,
            caption=message,
            parse_mode=constants.ParseMode.HTML 
        )
        st.toast("Telegram notification sent!", icon="🔔")
    except Exception as e:
        print(f"Failed to send Telegram notification: {e}")
        st.toast(f"Failed to send notification: {e}", icon="❌") 

# ---------------- End of TELEGRAM NOTIFICATION FUNCTION ----------------


# ---------------- Sidebar: configuration & model selection ----------------
st.sidebar.header("⚙️ Configuration")

# --- Load Models ---
@st.cache_resource
def load_all_models():
    """Caches both models to prevent reloading."""
    # model_main is for helmet, gloves, glasses, shoes
    model_main, labels_main = load_model("yolo9e.pt"), get_model_labels("yolo9e.pt")
    # model_vest is for vest only
    model_vest, labels_vest = load_model("best.pt"), get_model_labels("best.pt")
    return (model_main, labels_main), (model_vest, labels_vest)

try:
    (model_main, labels_main), (model_vest, labels_vest) = load_all_models()
    st.sidebar.success("✅ All models (yolo9e.pt, best.pt) loaded.")
except Exception as e:
    st.sidebar.error(f"Fatal: Failed loading models: {e}")
    st.stop()

# --- Hardcoded Detection Items ---
# All items that must be detected
FIXED_DETECTION_ITEMS = ["helmet", "vest", "gloves", "glasses", "shoes"]
# Items for the main model (yolo9e.pt)
ITEMS_MAIN_MODEL = ["helmet", "gloves", "glasses", "shoes"]
# Items for the vest model (best.pt)
ITEMS_VEST_MODEL = ["vest"]

st.sidebar.subheader("Detection Settings")
st.sidebar.info("Monitoring all required PPE: Helmet, Vest, Gloves, Glasses, Shoes.")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)


if st.sidebar.button("🔄 Clear Session & Reset"):
    st.session_state.clear()
    st.rerun()

# ---------------- Face registration UI (sidebar) ----------------
st.sidebar.subheader("🆔 Face Registration")
st.sidebar.info("Register employee face once. Stored locally.")

with st.sidebar.expander("Register New Employee"):
    emp_id = st.text_input("Employee ID (unique)", key="reg_empid")
    emp_name = st.text_input("Employee Name", key="reg_name")
    # If face_recognition not available, show message
    if not FACE_RECOGNITION_AVAILABLE:
        st.warning("Face recognition dependency missing. Install `face_recognition` (and its dlib dependency).")
        st.caption(f"Error: {_face_error}")
    else:
        reg_image = st.camera_input("Capture Face (clean, front-facing)")
        
        if reg_image and emp_id and emp_name:
            try:
                img = Image.open(reg_image).convert("RGB")
                img_np = np.array(img)
                encodings = face_recognition.face_encodings(img_np)
                if not encodings:
                    st.warning("No face detected. Try a clearer front-facing photo.")
                else:
                    encoding = encodings[0]
                    register_employee(emp_id.strip(), emp_name.strip(), np.array(encoding))
                    st.success(f"Registered {emp_name} ({emp_id}).")
            except Exception as e:
                st.error(f"Registration failed: {e}")
        elif reg_image:
             st.warning("Please enter Employee ID and Name to register.")

# Employee quick lookup
st.sidebar.subheader("🔎 Employee Quick Lookup")
lookup_empid = st.sidebar.text_input("Employee ID to lookup", key="lookup_empid")
if st.sidebar.button("Lookup Violations"):
    if lookup_empid:
        count = get_violation_count(lookup_empid.strip())
        st.sidebar.info(f"Total recorded violations for {lookup_empid.strip()}: {count}")
        all_hist = get_recent_violations(200)
        emp_hist = [h for h in all_hist if h['employee_id'] == lookup_empid.strip()]
        if emp_hist:
            for h in emp_hist[:10]:
                st.sidebar.write(f"{h['timestamp']}: {h['missing_items']}")
        else:
            st.sidebar.write("No recorded violations for this employee.")
    else:
        st.sidebar.warning("Enter an Employee ID.")

# ---------------- Helper UI functions ----------------
def display_metrics(person_count, total_violators, missing_counts, selected_items):
    """Displays the key metrics in columns."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("👥 Persons Detected", person_count)
    with col2:
        rate = (total_violators / person_count * 100) if person_count else 0
        st.metric("🚨 Violators", f"{total_violators} ({rate:.1f}%)")
    with col3:
        compliance = ((person_count - total_violators) / person_count * 100) if person_count else 100
        st.metric("✅ Compliance", f"{compliance:.1f}%")

    if total_violators:
        st.error("⚠️ **Safety Violations Detected**")
        for i in FIXED_DETECTION_ITEMS:
            if missing_counts.get(i, 0) > 0:
                st.markdown(f"- **{i}**: {missing_counts[i]} persons missing")
    elif person_count > 0:
        st.success("✅ **All Persons Compliant!**")
    else:
        st.info("No persons were detected in this analysis.")

def df_from_counts(counts: dict, selected_items: list) -> pd.DataFrame:
    """Creates a DataFrame from the missing counts dict for charting."""
    data = [{"PPE Item": k, "Missing Count": v} for k, v in counts.items() if k in FIXED_DETECTION_ITEMS and v > 0]
    return pd.DataFrame(data).sort_values("Missing Count", ascending=False) if data else pd.DataFrame()

# ---------------- Main layout: Tabs ----------------
tab1, tab2, tab3 = st.tabs(["📷 Image Analysis", "🎥 Video Analysis", "📜 History"])

# ---------------- Tab 1: Image Analysis ----------------
with tab1:
    st.header("📷 Image Analysis")

    # Live capture block
    st.markdown("### 🔴 Live Capture (Camera)")
    st.caption("Capture a live picture; the app will identify the user (if registered) and check PPE.")

    live_img = None
    if FACE_RECOGNITION_AVAILABLE:
        live_img = st.camera_input("Take a live photo")
    else:
        st.info("Face recognition disabled — live identification is not available.")

    if live_img:
        with st.spinner("Analyzing live capture..."):
            try:
                img_pil = Image.open(live_img).convert("RGB")
                img_np = np.array(img_pil)

                # Load registered encodings
                reg = load_all_encodings()  # dict: emp_id -> (name, enc)
                known_emp_ids = list(reg.keys())
                known_encodings = [reg[e][1] for e in known_emp_ids] if known_emp_ids else []
                known_names = [reg[e][0] for e in known_emp_ids] if known_emp_ids else []

                # --- NEW MULTI-PERSON FACE IDENTIFICATION ---
                # Find all faces and identify them
                face_locations = face_recognition.face_locations(img_np) if FACE_RECOGNITION_AVAILABLE else []
                face_encodings = face_recognition.face_encodings(img_np, face_locations) if FACE_RECOGNITION_AVAILABLE else []

                identified_persons = [] # Stores dicts: {"id":..., "name":..., "score":...}

                if face_encodings and known_encodings:
                    for unknown_encoding in face_encodings:
                        # Compare this unknown face to all known encodings
                        matches = face_recognition.compare_faces(known_encodings, unknown_encoding, tolerance=0.5)
                        face_distances = face_recognition.face_distance(known_encodings, unknown_encoding)
                        
                        matched_employee_id = None
                        matched_name = "Unknown"
                        match_score = None

                        # Find the best match (if any)
                        if True in matches:
                            best_idx = int(np.argmin(face_distances))
                            # Ensure the best match is within tolerance
                            if face_distances[best_idx] <= 0.5: 
                                matched_employee_id = known_emp_ids[best_idx]
                                matched_name = known_names[best_idx]
                                match_score = float(face_distances[best_idx])

                        identified_persons.append({
                            "id": matched_employee_id,
                            "name": matched_name,
                            "score": match_score
                        })
                elif face_encodings: # Faces detected, but no known encodings to compare against
                    for _ in face_encodings:
                        identified_persons.append({"id": None, "name": "Unknown", "score": None})

                # Get names for caption
                all_identified_names = [p["name"] for p in identified_persons]
                if not all_identified_names:
                    all_identified_names = ["No faces detected"]
                
                
                # --- MODIFIED FUNCTION CALL ---
                img_annot, missing_counts, violators, persons, detection_summary = detect_ppe_image(
                    img_pil,
                    required_items=FIXED_DETECTION_ITEMS,
                    confidence_threshold=confidence_threshold,
                    model_main=model_main,
                    model_main_items=ITEMS_MAIN_MODEL,
                    model_main_labels=labels_main,
                    model_vest=model_vest,
                    model_vest_items=ITEMS_VEST_MODEL,
                    model_vest_labels=labels_vest
                )

                # Save snapshot bytes (PNG) for history
                buf = io.BytesIO()
                img_annot.save(buf, format="PNG")
                img_bytes = buf.getvalue()

                # --- NEW MULTI-PERSON LOGGING AND UI ---
                # Determine missing items
                missing_list = [k for k, v in missing_counts.items() if v > 0]

                # Log violation and SEND NOTIFICATION if any missing items found
                if missing_list:
                    if identified_persons:
                        # Log a separate violation for EACH identified person
                        for person in identified_persons:
                            # NOTE: This logs the AGGREGATE missing list for each person.
                            log_violation(person["id"], person["name"], missing_list, img_bytes)
                            
                            # V V V FIX 3: USE asyncio.run() V V V
                            asyncio.run(
                                send_violation_notification(person["id"], person["name"], missing_list, img_bytes)
                            )
                            # ^ ^ ^ ^ ^ ^ ^ ^
                            
                    elif violators > 0:
                        # Log one violation for "Unknown" if no faces were identified
                        log_violation(None, "Unknown", missing_list, img_bytes)
                        
                        # V V V FIX 3: USE asyncio.run() V V V
                        asyncio.run(
                            send_violation_notification(None, "Unknown", missing_list, img_bytes)
                        )
                        # ^ ^ ^ ^ ^ ^ ^ ^


                # UI: show results
                caption_text = f"Identified: {', '.join(all_identified_names)}"
                st.image(img_annot, use_column_width=True, caption=caption_text)

                st.subheader("Detection Summary")
                display_metrics(persons, violators, missing_counts, FIXED_DETECTION_ITEMS)

                if identified_persons:
                    st.subheader("Identified Persons in Frame")
                    for person in identified_persons:
                        if person["id"]:
                            count = get_violation_count(person["id"])
                            score_txt = f"(score: {person['score']:.3f})" if person['score'] is not None else ""
                            st.info(f"✅ **{person['name']}** (ID: {person['id']}) {score_txt} — Previous violations: {count}")
                        else:
                            st.warning("⚠️ **Unknown** person detected. Consider registering them in the sidebar.")
                elif persons > 0:
                    st.info("Persons detected, but no registered faces were identified.")
                else:
                    st.info("No persons detected in this capture.")
            
            except Exception as e:
                st.error(f"Live capture error: {e}")
                st.exception(e) # Show full error

    st.markdown("---")

    # Existing upload flow
    uploaded_img = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    st.markdown("---")
    if st.button("🚀 Try Example Image (Demo)"):
        if not Path(EXAMPLE_IMAGE_PATH).exists():
            st.error(f"⚠️ Example file not found! The demo button needs `{EXAMPLE_IMAGE_PATH}` in the root folder.")
        else:
            uploaded_img = EXAMPLE_IMAGE_PATH
            try:
                st.toast("Example image loaded!", icon='🚀')
            except Exception:
                pass

    if uploaded_img:
        with st.spinner("🔍 Analyzing image..."):
            try:
                # --- MODIFIED FUNCTION CALL ---
                img, missing, violators, persons, _ = detect_ppe_image(
                    uploaded_img,
                    required_items=FIXED_DETECTION_ITEMS,
                    confidence_threshold=confidence_threshold,
                    model_main=model_main,
                    model_main_items=ITEMS_MAIN_MODEL,
                    model_main_labels=labels_main,
                    model_vest=model_vest,
                    model_vest_items=ITEMS_VEST_MODEL,
                    model_vest_labels=labels_vest
                )

                col1, col2 = st.columns([0.65, 0.35])
                with col1:
                    st.image(img, caption="🎯 Detection Result", use_column_width=True)
                with col2:
                    st.subheader("Analysis Results")
                    display_metrics(persons, violators, missing, FIXED_DETECTION_ITEMS)
                    df = df_from_counts(missing, FIXED_DETECTION_ITEMS)
                    if not df.empty:
                        st.markdown("---")
                        st.subheader("Missing PPE Breakdown")
                        chart = alt.Chart(df).mark_bar().encode(
                            x=alt.X('Missing Count:Q', title='Number of Persons Missing Item'),
                            y=alt.Y('PPE Item:N', sort='-x'),
                            tooltip=['PPE Item', 'Missing Count']
                        ).interactive()
                        st.altair_chart(chart, use_container_width=True)
            except Exception as e:
                st.error(f"An error occurred during image processing: {e}")
                st.exception(e) # Show full error

# ---------------- Tab 2: Video Analysis ----------------
with tab2:
    st.header("🎥 Video Analysis")
    uploaded_vid = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

    if uploaded_vid:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_vid.read())
                input_path = tmp.name

            progress = st.progress(0, text="Starting video processing...")

            def update_prog(p):
                try:
                    progress.progress(p / 100.0, text=f"Processing... {int(p)}%")
                except Exception:
                    pass

            with st.spinner(f"Analyzing video (this may take a while)..."):
                
                # --- MODIFIED FUNCTION CALL ---
                out_path, missing, violators, persons, _ = detect_ppe_video(
                    input_path, "output.mp4",
                    required_items=FIXED_DETECTION_ITEMS,
                    confidence_threshold=confidence_threshold,
                    progress_callback=update_prog,
                    model_main=model_main,
                    model_main_items=ITEMS_MAIN_MODEL,
                    model_main_labels=labels_main,
                    model_vest=model_vest,
                    model_vest_items=ITEMS_VEST_MODEL,
                    model_vest_labels=labels_vest
                )

            progress.empty()
            st.video(out_path)

            st.subheader("Video Analysis Results")
            st.info("Metrics below are aggregated for the entire video.")
            display_metrics(persons, violators, missing, FIXED_DETECTION_ITEMS)

        except Exception as e:
            st.error(f"An error occurred during video processing: {e}")
            st.exception(e) # Show full error
            if 'progress' in locals():
                progress.empty()
        finally:
            if 'input_path' in locals() and Path(input_path).exists():
                try:
                    Path(input_path).unlink()
                except Exception:
                    pass

# ---------------- Tab 3: History ----------------
with tab3:
    st.header("📜 Violation History (recent)")
    st.caption("Shows recent violations (who, when, what was missing). Uses local SQLite storage.")

    hist_rows = get_recent_violations(limit=200)
    if not hist_rows:
        st.info("No violation history yet.")
    else:
        # Present as a table with expanders for images/details
        df_list = []
        for row in hist_rows:
            df_list.append({
                "id": row['id'],
                "employee_id": row['employee_id'] or "Unknown",
                "name": row['name'] or "Unknown",
                "timestamp": row['timestamp'],
                "missing_items": ", ".join(row['missing_items'])
            })
        history_df = pd.DataFrame(df_list)
        st.dataframe(history_df.sort_values("timestamp", ascending=False))

        st.markdown("---")
        st.subheader("Recent entries (with snapshots)")
        for row in hist_rows[:50]:
            ts = row['timestamp'] or ""
            name = row['name'] or "Unknown"
            emp_id_row = row['employee_id'] or "Unknown"
            missing_str = ", ".join(row['missing_items']) if row['missing_items'] else "None"
            col1, col2 = st.columns([0.7, 0.3])
            with col1:
                st.markdown(f"**{name}** — ID: `{emp_id_row}`")
                st.caption(ts.replace("T", " ") if ts else "")
                st.markdown(f"**Missing:** {missing_str}")
            with col2:
                if row['image_blob']:
                    try:
                        st.image(row['image_blob'], width=220)
                    except Exception:
                        st.text("Image cannot be displayed.")
            st.markdown("---")

# ---------------- Footer / Notes ----------------
st.markdown("---")
st.markdown(
    """
    Built with ❤️ using YOLO & Streamlit | [@Ambuj-N](https://github.com/Ambuj-N) | IIT BHU 🎓
    """
)