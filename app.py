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
import base64
from typing import Optional, List 
import asyncio 
import requests
import telegram 

# --- UTILITY FUNCTIONS ---

def get_image_as_base64(path):
    try:
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return None

try:
    from telegram import constants
except ImportError:
    class DummyConstants:
        class ParseMode:
            HTML = 'HTML'
    constants = DummyConstants()

try:
    if not hasattr(telegram, 'Bot'):
        class DummyBot:
            def __init__(self, *args, **kwargs): pass
            async def send_photo(self, *args, **kwargs): raise RuntimeError("Telegram not installed.")
        telegram.Bot = DummyBot
except Exception:
    pass

try:
    from utils.detect import detect_ppe_image, detect_ppe_video, load_model, get_model_labels
except ImportError:
    st.error("Fatal Error: `utils/detect.py` not found. Please ensure it's in the same directory.")
    st.stop()


FACE_RECOGNITION_AVAILABLE = True
try:
    import face_recognition
except Exception as e:
    FACE_RECOGNITION_AVAILABLE = False
    _face_error = e


try:
    from utils.face_db import (
        register_employee,
        load_all_encodings,
        log_violation,
        get_violation_count,
        get_recent_violations,
        delete_violation_entry, 
        delete_employee
    )
except ImportError:
    st.error("Fatal Error: `utils/face_db.py` not found. Please create it (see instructions).")
    st.stop()
except NameError:
    def delete_violation_entry(id): st.warning(f"DB stub: Delete violation {id} logic not implemented.")
    def delete_employee(emp_id): st.warning(f"DB stub: Delete employee {emp_id} logic not implemented.")
    st.warning("⚠️ Warning: Admin delete functions are not yet defined in `utils/face_db.py`.")


EXAMPLE_IMAGE_PATH = "example_ppe.jpg"

st.set_page_config(
    page_title="APECS",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="logo.png"
)


logo_base64 = get_image_as_base64("logo.png")


if logo_base64:
    st.markdown(f"""
    <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 2rem; margin-top: 25px;">
        <img src="data:image/png;base64,{logo_base64}" alt="APECS Logo" style="height: 200px; margin-right: 20px; transform: translateY(10px); ">
        <div>
            <h1 style="font-weight:bold; color:#FFFFFF; font-size:3rem; margin: 0;">APECS</h1>
            <h3 style="color:#FFFFF; margin: 0;">Automated PPE Enforcement & Compliance System</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 2rem;">
        <span style="font-size: 4rem; margin-right: 20px;">🦺</span>
        <div>
            <h1 style="font-weight:bold; color:#2E86C1; font-size:3rem; margin: 0;">APECS</h1>
            <h3 style="color:#5D6D7E; margin: 0;">Automated PPE Enforcement & Compliance System</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)

with st.expander("👋 Welcome! Meet the Development Team", expanded=False):
    st.markdown("""
    **Welcome to APECS!** This app uses AI to monitor workplace safety.

    Developed by students from **IIT (BHU) Varanasi**:
    - **Ambuj Nayak** ([GitHub](https://github.com/Ambuj-N))
    - **Paturi Hemanth Sai**
    - **Ankit Raj**
    - **Jalla Poojitha**
    """)


# --- TELEGRAM SETUP ---

BOT_TOKEN = "8258711886:AAGUwUmWsyrfHWpAWnhgieEL9ESobk_NsAs"
CHAT_ID = "1269174608"

async def send_violation_notification(emp_id: Optional[str], name: Optional[str], missing_items: List[str], img_bytes: bytes):
    """Sends a violation alert to a Telegram chat."""
    
    if not hasattr(telegram, 'Bot') or not hasattr(constants.ParseMode, 'HTML'):
        st.toast("Telegram library not properly initialized/installed.", icon="❌")
        return

    name_str = name or "Unknown"
    id_str = emp_id or "N/A"
    missing_str = ", ".join(missing_items)
    
    
    message = (
        f"<b>🚨 PPE VIOLATION DETECTED 🚨</b>\n\n"
        f"<b>Name:</b> {name_str}\n"
        f"<b>ID:</b> {id_str}\n"
        f"<b>Missing:</b> {missing_str}"
    )
    
    try:
        bot = telegram.Bot(token=BOT_TOKEN)
        
        
        image_file = io.BytesIO(img_bytes)
        image_file.name = "violation.png"
        image_file.seek(0)
        
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


# --- MODEL LOADING ---

@st.cache_resource
def load_all_models():
    """Caches both models to prevent reloading."""
    model_main, labels_main = load_model("yolo9e.pt"), get_model_labels("yolo9e.pt")
    model_vest, labels_vest = load_model("best.pt"), get_model_labels("best.pt")
    return (model_main, labels_main), (model_vest, labels_vest)

try:
    (model_main, labels_main), (model_vest, labels_vest) = load_all_models()
except Exception as e:
    st.sidebar.header("⚠️ Fatal Error")
    st.sidebar.error(f"Failed loading models (yolo9e.pt, best.pt): {e}. Application cannot start.")
    st.stop()


# --- SIDEBAR CONFIGURATION ---

st.sidebar.header("⚙️ Configuration") 

FIXED_DETECTION_ITEMS = ["helmet", "vest", "gloves", "glasses", "shoes"]
ITEMS_MAIN_MODEL = ["helmet", "gloves", "glasses", "shoes"]
ITEMS_VEST_MODEL = ["vest"]

st.sidebar.subheader("Detection Settings")
st.sidebar.info("Monitoring all required PPE: Helmet, Vest, Gloves, Glasses, Shoes.")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)


if st.sidebar.button("🔄 Clear Session & Reset"):
    st.session_state.clear()
    st.experimental_rerun() 


st.sidebar.subheader("🆔 Face Registration")
st.sidebar.info("Register employee face once. Stored locally.")

with st.sidebar.expander("Register New Employee"):
    emp_id = st.text_input("Employee ID (unique)", key="reg_empid")
    emp_name = st.text_input("Employee Name", key="reg_name")
    
    if not FACE_RECOGNITION_AVAILABLE:
        st.warning(f"Face recognition disabled: {_face_error}. Please install `face_recognition` and its dlib dependency if running locally.")
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


st.sidebar.subheader("🔎 Employee Quick Lookup")
lookup_query = st.sidebar.text_input("Search by ID or Name", key="lookup_query")
if st.sidebar.button("Lookup Violations"):
    query = lookup_query.strip()
    if query:
        
        reg_data = load_all_encodings()
        
        target_id = None
        target_name = None
        
        
        if query in reg_data:
            target_id = query
            target_name = reg_data[query][0]
        
        else:
            for emp_id, (name, _) in reg_data.items():
                if name.strip().lower() == query.lower():
                    target_id = emp_id
                    target_name = name
                    break

        if target_id or (target_id is None and query.lower() == "unknown"): 
            
            search_id = target_id if target_id else None
            
            count = get_violation_count(search_id)
            lookup_label = target_name if target_name else query
            
            st.sidebar.info(f"Total recorded violations for **{lookup_label}** (ID: {search_id or 'N/A'}): **{count}**")
            
            all_hist = get_recent_violations(200)
            emp_hist = [h for h in all_hist if (h['employee_id'] == search_id and h['employee_id'] is not None) or (h['employee_id'] is None and search_id is None)]
            
            if emp_hist:
                st.sidebar.write("Recent Violations:")
                for h in emp_hist[:10]: 
                    ts = h['timestamp'].split('T')[0] if h['timestamp'] else 'N/A'
                    st.sidebar.markdown(f"• {ts}: {', '.join(h['missing_items'])}")
            else:
                st.sidebar.write("No recorded violations found.")
        else:
            st.sidebar.warning(f"Employee ID or Name '{query}' not found in registry.")
    else:
        st.sidebar.warning("Enter an Employee ID or Name.")


# --- METRICS AND CHARTING FUNCTIONS ---

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


# --- TAB DEFINITION (MODIFIED) ---

tab1, tab2, tab3, tab_history, tab_admin = st.tabs([
    "📷 Image Analysis", 
    "🎥 Video Analysis",
    "📊 Analytics & Documentation",
    "📜 History Log", # Public access tab
    "🔑 Admin Panel" # Restricted access tab
])


# --- IMAGE ANALYSIS TAB (UNCHANGED) ---

with tab1:
    st.header("📷 Image Analysis")

    
    st.markdown("### 🔴 Live Capture (Camera)")
    st.caption("Capture a live picture; the app will identify the user (if registered) and check PPE.")

    live_img = st.camera_input("Take a live photo") 

    if not FACE_RECOGNITION_AVAILABLE:
        st.info("⚠️ **Face Identification is Disabled.** The app can still capture and check PPE, but it cannot identify registered employees or log by name because the `face_recognition` dependency could not load on the web platform.")

    if live_img:
        with st.spinner("Analyzing live capture..."):
            try:
                img_pil = Image.open(live_img).convert("RGB")
                img_np = np.array(img_pil)

                
                reg = load_all_encodings()  # dict: emp_id -> (name, enc)
                known_emp_ids = list(reg.keys())
                known_encodings = [reg[e][1] for e in known_emp_ids] if known_emp_ids else []
                known_names = [reg[e][0] for e in known_emp_ids] if known_emp_ids else []

                
                if FACE_RECOGNITION_AVAILABLE:
                    face_locations = face_recognition.face_locations(img_np)
                    face_encodings = face_recognition.face_encodings(img_np, face_locations)
                else:
                    face_locations = []
                    face_encodings = []

                identified_persons = [] # Stores dicts: {"id":..., "name":..., "score":...}

                if face_encodings and known_encodings:
                    for unknown_encoding in face_encodings:
                        matches = face_recognition.compare_faces(known_encodings, unknown_encoding, tolerance=0.5)
                        face_distances = face_recognition.face_distance(known_encodings, unknown_encoding)
                        
                        matched_employee_id = None
                        matched_name = "Unknown"
                        match_score = None

                        if True in matches:
                            best_idx = int(np.argmin(face_distances))
                            if face_distances[best_idx] <= 0.5: 
                                matched_employee_id = known_emp_ids[best_idx]
                                matched_name = known_names[best_idx]
                                match_score = float(face_distances[best_idx])

                        identified_persons.append({
                            "id": matched_employee_id,
                            "name": matched_name,
                            "score": match_score
                        })
                elif face_encodings: 
                    for _ in face_encodings:
                        identified_persons.append({"id": None, "name": "Unknown", "score": None})

                all_identified_names = [p["name"] for p in identified_persons]
                if not all_identified_names:
                    all_identified_names = ["No faces detected"]
                
                
                
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

                
                buf = io.BytesIO()
                img_annot.save(buf, format="PNG")
                img_bytes = buf.getvalue()

                
                missing_list = [k for k, v in missing_counts.items() if v > 0]

                if missing_list:
                    if identified_persons:
                        for person in identified_persons:
                            log_violation(person["id"], person["name"], missing_list, img_bytes)
                            
                            asyncio.run(
                                send_violation_notification(person["id"], person["name"], missing_list, img_bytes)
                            )
                            
                    elif violators > 0:
                        log_violation(None, "Unknown", missing_list, img_bytes)
                        
                        asyncio.run(
                            send_violation_notification(None, "Unknown", missing_list, img_bytes)
                        )

                
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
                st.exception(e)

    st.markdown("---")

    
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
                st.exception(e)

# --- VIDEO ANALYSIS TAB (UNCHANGED) ---

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
            st.exception(e)
            if 'input_path' in locals():
                progress.empty()
        finally:
            if 'input_path' in locals() and Path(input_path).exists():
                try:
                    Path(input_path).unlink()
                except Exception:
                    pass

# --- ANALYTICS TAB (UNCHANGED) ---

with tab3:
    st.markdown('<div class="section-header">📊 Analytics & Documentation</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 How It Works
        
        1. **🆔 Register Employee**: Use the sidebar to register an employee with their ID, name, and face.
        2. **📸 Capture or Upload**: Use the live camera or upload an image/video.
        3. **🔍 AI Analysis**: The system performs two key tasks:
            - **Identifies** the person using face recognition.
            - **Detects** all required PPE items.
        4. **🚨 Log & Alert**: If a violation occurs:
            - It's logged in the **History Log** tab with a snapshot.
            - An instant **Telegram notification** is sent.
        5. **📊 Review**: Check the dashboard for compliance rates and view detailed violation history.
        
        ### 🛡️ Monitored PPE Items
        
        This system automatically monitors all of the following PPE:
        """)
        for item in FIXED_DETECTION_ITEMS:
            st.markdown(f"- 🔹 **{item.title()}**")
    
    with col2:
        st.markdown("""
        ### ⚙️ Technology Stack
        
        - **PPE Models**: YOLOv9e (General) & Custom YOLO (Vests)
        - **Face Recognition**: `face_recognition` (dlib)
        - **Framework**: Streamlit & Ultralytics
        - **Database**: Local SQLite for history (See Admin tab for security notes)
        - **Notifications**: Telegram Bot API
        
        ### 🎨 Core Features
        
        - ✨ **Real-time** detection via camera
        - 🆔 **Employee Identification** via face recognition
        - 🔔 **Instant Alerts** on Telegram for violations
        - 📜 **Persistent History** with violation snapshots
        - 👥 **Multi-person** detection and identification
        - 📊 Detailed compliance metrics
        
        ### 💡 Tips
        
        - For best face recognition, register users with clear, front-facing photos.
        - The confidence threshold affects PPE detection; higher values reduce false positives.
        - Check the **Admin Panel** tab for full audit logs and data management.
        """)
    
    st.markdown("---")
    st.markdown("""
    ### 🧑‍💻 Development Team
    
    **Created by:**
    - **Ambuj Nayak** - [GitHub](https://github.com/Ambuj-N) - [24074007]
    - **Paturi Hemanth Sai** - [24075065]
    - **Ankit Raj** - [24074011]
    - **Jalla Poojitha** - [24124022]
    
    **Institution**: Indian Institute of Technology (BHU) Varanasi
    
    **Purpose**: Enhancing workplace safety through AI-powered PPE compliance monitoring.
    """)


# --- ADMIN CREDENTIAL SETUP ---

try:
    ADMIN_USER = st.secrets["admin_credentials"]["username"]
    ADMIN_PASS = st.secrets["admin_credentials"]["password"]
except KeyError:
    st.error("FATAL ERROR: Admin credentials not found in `.streamlit/secrets.toml`. Please create and format it correctly.")
    ADMIN_USER = ""
    ADMIN_PASS = ""

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False


# --- HISTORY LOG (PUBLIC ACCESS) ---

with tab_history:
    st.header("📜 History Log")

    # --- VIOLATION HISTORY DISPLAY (PUBLIC READ) ---
    st.markdown("## Violation History Audit Log")
    st.caption("Shows all recent violations with snapshots. **Deletion is restricted to Admin users.**")

    hist_rows = get_recent_violations(limit=200)
    
    if not hist_rows:
        st.info("No violation history yet.")
    else:
        # Present as a table for quick overview
        df_list = []
        for row in hist_rows:
            df_list.append({
                "ID": row['id'],
                "Employee ID": row['employee_id'] or "Unknown",
                "Name": row['name'] or "Unknown",
                "Timestamp": row['timestamp'],
                "Missing Items": ", ".join(row['missing_items'])
            })
        history_df = pd.DataFrame(df_list)
        st.dataframe(history_df.sort_values("Timestamp", ascending=False))

        st.markdown("---")
        st.subheader("Detailed Entries (with Snapshots)")
        
        # Display detailed entries with images and conditional delete buttons
        for row in hist_rows[:50]:
            col1, col2, col3 = st.columns([0.5, 0.3, 0.2])
            
            ts = row['timestamp'] or ""
            name = row['name'] or "Unknown"
            emp_id_row = row['employee_id'] or "Unknown"
            missing_str = ", ".join(row['missing_items']) if row['missing_items'] else "None"
            
            with col1:
                st.markdown(f"**{name}** — ID: `{emp_id_row}` (DB ID: `{row['id']}`)")
                st.caption(ts.replace("T", " ") if ts else "")
                st.markdown(f"**Missing:** {missing_str}")
                
            with col2:
                if row['image_blob']:
                    try:
                        st.image(row['image_blob'], width=220, caption="Violation Snapshot")
                    except Exception:
                        st.text("Image cannot be displayed.")
                else:
                     st.text("No image saved.")
            
            with col3:
                # --- ACCESS CONTROL FOR DELETION ---
                is_admin = st.session_state.authenticated
                if is_admin:
                    if st.button("Delete Entry", key=f"del_viol_{row['id']}"):
                        delete_violation_entry(row['id'])
                        st.toast(f"Violation ID {row['id']} deleted!", icon="🗑️")
                        st.experimental_rerun()
                else:
                    # Non-admin users see a disabled-style button and get a warning on click
                    if st.button("Delete Entry (Admin Only)", key=f"del_viol_{row['id']}_public"):
                        st.warning("🚨 **Admin Access Required.** Please use the **Admin Panel** tab to log in and gain deletion permissions.")
                    
            st.markdown("---")

    # --- EMPLOYEE DATABASE (PUBLIC READ) ---
    st.markdown("## 🆔 Registered Employee Database")
    st.caption("View registered employees. Deletion is restricted to Admin users.")

    emp_data = load_all_encodings() # dict: emp_id -> (name, enc)
    
    if not emp_data:
        st.info("No employees are currently registered.")
    else:
        emp_df_list = []
        for emp_id, (name, _) in emp_data.items():
            emp_df_list.append({"Employee ID": emp_id, "Name": name})
        
        emp_df = pd.DataFrame(emp_df_list)
        st.dataframe(emp_df)


# --- ADMIN PANEL (RESTRICTED ACCESS) ---

with tab_admin:
    st.header("🔑 Admin Panel")
    st.caption("Full control over database management, including deletions.")

    col_login, col_controls = st.columns([0.3, 0.7])
    
    with col_login:
        st.markdown("### Admin Login")
        if st.session_state.authenticated:
            st.success(f"Welcome, {ADMIN_USER}! You have full control.")
            if st.button("Logout", key="admin_logout_btn"):
                st.session_state.authenticated = False
                st.experimental_rerun()
        else:
            with st.form("admin_login_form_admin_tab", clear_on_submit=False):
                username = st.text_input("Username", key="login_user_tab_admin")
                password = st.text_input("Password", type="password", key="login_pass_tab_admin")
                submitted = st.form_submit_button("Login")
                
                if submitted:
                    if username == ADMIN_USER and password == ADMIN_PASS and ADMIN_USER and ADMIN_PASS:
                        st.session_state.authenticated = True
                        st.success("Logged in successfully!")
                        st.experimental_rerun()
                    else:
                        st.error("Invalid Username or Password")
            
            st.info("Hint: Credentials are read from `.streamlit/secrets.toml`.")


    with col_controls:
        if not st.session_state.authenticated:
            st.warning("Log in to the left to access data management controls.")
        else:
            # --- EMPLOYEE DATABASE MANAGEMENT ---
            st.markdown("## 🗑️ Delete Employee Records")
            st.caption("Permanently remove employee records (name, ID, face encoding). Violation history remains.")

            emp_data = load_all_encodings() 
            
            if not emp_data:
                st.info("No employees are currently registered to delete.")
            else:
                for emp_id, (name, _) in emp_data.items():
                    col_emp1, col_emp2, col_emp3 = st.columns([0.3, 0.5, 0.2])
                    with col_emp1:
                        st.text(f"ID: {emp_id}")
                    with col_emp2:
                        st.text(f"Name: {name}")
                    with col_emp3:
                        if st.button("Delete Employee", key=f"del_emp_{emp_id}"):
                            delete_employee(emp_id)
                            st.toast(f"Employee {emp_id} deleted!", icon="🗑️")
                            st.experimental_rerun()
                
# ---------------- Footer / Notes ----------------
st.markdown("---")
st.markdown(
    """
    Built with ❤️ using YOLO & Streamlit | [@Ambuj-N](https://github.com/Ambuj-N) | IIT BHU 🎓 | APECS
    """
)
