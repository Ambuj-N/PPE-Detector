from pathlib import Path
from typing import Tuple, Dict, List, Optional, Callable
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO

# Optional HF download
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

# Hugging Face repository configuration
HF_MODEL_REPO = "Anbhigya/ppe-detector-model"

# Model registry and cache
_LOADED_MODELS: Dict[str, YOLO] = {}
_MODEL_LABELS: Dict[str, Dict[int, str]] = {}

# Model files mapping: model_name -> HF filename
MODEL_FILES = {
    "yolo9e.pt": "yolo9e.pt",
    "best.pt": "best.pt",
    "good.pt": "good.pt"
}

# ----------------------
# MODEL MANAGEMENT
# ----------------------
def load_model(model_name: str) -> YOLO:
    """
    Load a YOLO model dynamically and cache it.
    First checks for local file, then downloads from Hugging Face if needed.
    """
    global _LOADED_MODELS, _MODEL_LABELS
    
    if model_name in _LOADED_MODELS:
        return _LOADED_MODELS[model_name]

    if model_name not in MODEL_FILES:
        raise ValueError(f"Model {model_name} not found in MODEL_FILES! Check utils/detect.py")

    local_path = Path(MODEL_FILES[model_name])
    if local_path.exists():
        model_path = str(local_path)
        print(f"✅ Loading model from local path: {model_path}")
    else:
        if hf_hub_download is None:
            raise RuntimeError(
                "huggingface_hub is not installed and local model not found! "
                "Install it with: pip install huggingface_hub"
            )
        print(f"⬇️  Downloading model '{model_name}' from Hugging Face repo: {HF_MODEL_REPO}")
        model_path = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=MODEL_FILES[model_name]
        )
        print(f"✅ Model downloaded to: {model_path}")

    try:
        model = YOLO(model_path)
    except ModuleNotFoundError as e:
        if "ultralytics.yolo" in str(e):
            print(f"⚠️  Model uses older ultralytics format, applying compatibility fix...")
            import torch
            import sys
            
            if 'ultralytics.yolo' not in sys.modules:
                import ultralytics
                sys.modules['ultralytics.yolo'] = ultralytics
                sys.modules['ultralytics.yolo.utils'] = ultralytics.utils
                sys.modules['ultralytics.yolo.v8'] = ultralytics
            
            model = YOLO(model_path)
            print(f"✅ Loaded model with compatibility fix")
        else:
            raise
    
    _LOADED_MODELS[model_name] = model
    _MODEL_LABELS[model_name] = extract_labels_from_model(model, model_name)
    return model

def extract_labels_from_model(model: YOLO, model_name: str) -> Dict[int, str]:
    """Extract class labels directly from the loaded YOLO model."""
    try:
        if hasattr(model, 'names'):
            names = model.names
            if isinstance(names, dict):
                return names
            elif isinstance(names, list):
                return {i: name for i, name in enumerate(names)}
        
        if hasattr(model, 'model') and hasattr(model.model, 'names'):
            names = model.model.names
            if isinstance(names, dict):
                return names
            elif isinstance(names, list):
                return {i: name for i, name in enumerate(names)}
        
        print(f"⚠️  Could not extract labels from model, using fallback for {model_name}")
    except Exception as e:
        print(f"⚠️  Error extracting labels: {e}, using fallback for {model_name}")
    
    return get_fallback_labels(model_name)

def get_fallback_labels(model_name: str) -> Dict[int, str]:
    """Fallback label mappings if dynamic extraction fails."""
    if model_name == "yolo9e.pt":
        return {
            0: "Person", 1: "Helmet", 2: "Gloves", 3: "Safety-vest",
            4: "Face-mask-medical", 5: "Earmuffs", 6: "Shoes", 7: "glasses" 
        }
    elif model_name == "best.pt":
        # Assuming best.pt is specialized for vest (using a common label for vest)
        return {
            0: "Person", 1: "vest" 
        }
    else:
        # Generic fallback
        return {0: "Person"}

def get_model_labels(model_name: str) -> Dict[int, str]:
    """Return the label mapping for a specific model."""
    if model_name not in _MODEL_LABELS:
        load_model(model_name)
    return _MODEL_LABELS[model_name]

# ----------------------
# HELPER FUNCTIONS
# ----------------------
def _normalize(s: str) -> str:
    return s.strip().lower()

def _matches_ppe_item(selected_item: str, detected_label: str) -> bool:
    """Check if a selected PPE item matches a detected label."""
    selected_norm = selected_item.lower().replace('-', ' ').replace('_', ' ')
    detected_norm = detected_label.lower().replace('-', ' ').replace('_', ' ')
    
    if selected_norm == detected_norm:
        return True
    
    if selected_norm in detected_norm or detected_norm in selected_norm:
        return True
    
    if 'vest' in selected_norm and 'vest' in detected_norm:
        return True
    
    if 'mask' in selected_norm and 'mask' in detected_norm:
        return True
    
    return False

def _box_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return (float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0

def _point_in_box(px, py, box):
    x1, y1, x2, y2 = box
    return (px >= x1) and (px <= x2) and (py >= y1) and (py <= y2)

def _extract_detections(
    res, 
    model_labels: Dict[int, str], 
    confidence_threshold: float
) -> List[Tuple[str, np.ndarray, float]]:
    """Helper to extract boxes, labels, and scores from a results object."""
    detections = []
    boxes = getattr(res, "boxes", None)
    if boxes is None:
        return detections

    try:
        xyxy_arr = boxes.xyxy.cpu().numpy()
        cls_arr = boxes.cls.cpu().numpy().astype(int)
        conf_arr = boxes.conf.cpu().numpy()
    except Exception:
        data = boxes.data.cpu().numpy()
        xyxy_arr = data[:, :4]
        cls_arr = data[:, -1].astype(int)
        conf_arr = data[:, 4] if data.shape[1] >= 5 else np.ones(len(cls_arr))

    for i, cls_idx in enumerate(cls_arr):
        confidence = float(conf_arr[i])
        if confidence < confidence_threshold:
            continue
        
        label = model_labels.get(int(cls_idx), str(int(cls_idx)))
        xyxy = xyxy_arr[i].astype(float)
        detections.append((label, xyxy, confidence))
    
    return detections

# ----------------------
# IMAGE DETECTION (MODIFIED & FIXED)
# ----------------------
def detect_ppe_image(
    uploaded_file_or_pil,
    required_items: List[str],
    confidence_threshold: float,
    model_main: YOLO,
    model_main_items: List[str],
    model_main_labels: Dict[int, str],
    model_vest: YOLO,
    model_vest_items: List[str],
    model_vest_labels: Dict[int, str]
) -> Tuple[Image.Image, Dict[str, int], int, int, Dict[str, int]]:
    """
    Detect PPE in a single image using two separate YOLO models.
    """
    # Convert to PIL
    if hasattr(uploaded_file_or_pil, "read"):
        uploaded_file_or_pil.seek(0)
        pil = Image.open(uploaded_file_or_pil).convert("RGB")
    elif isinstance(uploaded_file_or_pil, Image.Image):
        pil = uploaded_file_or_pil.convert("RGB")
    else:
        pil = Image.open(uploaded_file_or_pil).convert("RGB")

    img_np = np.array(pil)

    # --- Run Detection on Both Models ---
    results_main = model_main(img_np, conf=confidence_threshold)
    results_vest = model_vest(img_np, conf=confidence_threshold)

    # --- Extract and Filter Detections ---
    dets_main = _extract_detections(results_main[0], model_main_labels, confidence_threshold)
    dets_vest = _extract_detections(results_vest[0], model_vest_labels, confidence_threshold)
    
    persons = []
    all_other_dets = []

    # Process Main Model Detections (Persons + PPE)
    for label, xyxy, conf in dets_main:
        if _normalize(label) == "person":
            persons.append((xyxy, conf))
        else:
            for item_name in model_main_items:
                if _matches_ppe_item(item_name, label):
                    all_other_dets.append((item_name, xyxy, conf))
                    break
    
    # Process Vest Model Detections (PPE only)
    for label, xyxy, conf in dets_vest:
        if _normalize(label) == "person":
            continue
        
        for item_name in model_vest_items:
            if _matches_ppe_item(item_name, label):
                all_other_dets.append((item_name, xyxy, conf))
                break

    # --- Violation Checking ---
    person_count = len(persons)
    missing_counts = {it: 0 for it in required_items}
    violator_flags = [False] * max(0, person_count)
    
    detection_summary = {label: sum(1 for l, *_ in all_other_dets if l==label) for label in required_items}

    for p_idx, (p_box, _) in enumerate(persons):
        px1, py1, px2, py2 = p_box
        for warning_item in required_items:
            present = any(
                warning_item == label and _point_in_box(*_box_center(b), (px1, py1, px2, py2))
                for label, b, _ in all_other_dets 
            )
            if not present:
                missing_counts[warning_item] += 1
                violator_flags[p_idx] = True

    total_violators = sum(1 for v in violator_flags if v)

    # --- Annotate Image ---
    # 💥 FIX APPLIED HERE 💥
    annotated_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Draw all detected PPE (from combined list)
    for label, b, confidence in all_other_dets:
        color = (0, 255, 0) # Green for detected PPE
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(annotated_bgr, (x1,y1), (x2,y2), color, 2)
        cv2.putText(annotated_bgr, f"{label} {confidence:.2f}", (x1,max(16,y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Draw person boxes
    for p_idx, (p_box, confidence) in enumerate(persons):
        x1, y1, x2, y2 = map(int, p_box)
        compliant = not violator_flags[p_idx]
        color = (0,255,0) if compliant else (0,0,255) # Green for compliant, Red for violator
        cv2.rectangle(annotated_bgr, (x1,y1),(x2,y2), color, 3)
        status = "COMPLIANT" if compliant else "VIOLATOR"
        cv2.putText(annotated_bgr, f"Person {status} {confidence:.2f}", (x1,max(16,y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated_rgb), missing_counts, total_violators, person_count, detection_summary

# ----------------------
# VIDEO DETECTION (Checked and correct)
# ----------------------
def detect_ppe_video(
    input_video_path: str,
    output_video_path: str,
    required_items: List[str],
    confidence_threshold: float,
    progress_callback: Optional[Callable[[float], None]],
    model_main: YOLO,
    model_main_items: List[str],
    model_main_labels: Dict[int, str],
    model_vest: YOLO,
    model_vest_items: List[str],
    model_vest_labels: Dict[int, str]
) -> Tuple[str, Dict[str, int], int, int, Dict[str, int]]:
    """
    Detect PPE frame-by-frame in video using two separate YOLO models.
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open input video")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    missing_counts = {it: 0 for it in required_items}
    violator_events = 0
    persons_seen = 0
    detection_summary = {}

    frame_count = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_count += 1
        if progress_callback and total_frames > 0:
            progress_callback(int((frame_count / total_frames) * 100))

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # --- Run Detection on Both Models ---
        results_main = model_main(frame_rgb, conf=confidence_threshold)
        results_vest = model_vest(frame_rgb, conf=confidence_threshold)

        # --- Extract and Filter Detections ---
        dets_main = _extract_detections(results_main[0], model_main_labels, confidence_threshold)
        dets_vest = _extract_detections(results_vest[0], model_vest_labels, confidence_threshold)
        
        persons = []
        all_other_dets = []

        # Process Main Model
        for label, xyxy, conf in dets_main:
            if _normalize(label) == "person":
                persons.append((xyxy, conf))
            else:
                for item_name in model_main_items:
                    if _matches_ppe_item(item_name, label):
                        all_other_dets.append((item_name, xyxy, conf))
                        detection_summary[item_name] = detection_summary.get(item_name, 0) + 1
                        break
        
        # Process Vest Model
        for label, xyxy, conf in dets_vest:
            if _normalize(label) == "person":
                continue
            for item_name in model_vest_items:
                if _matches_ppe_item(item_name, label):
                    all_other_dets.append((item_name, xyxy, conf))
                    detection_summary[item_name] = detection_summary.get(item_name, 0) + 1
                    break

        # --- Violation Checking (for this frame) ---
        persons_seen += len(persons)
        violator_flags = [False] * max(0, len(persons))

        for p_idx, (p_box, _) in enumerate(persons):
            px1, py1, px2, py2 = p_box
            for warning_item in required_items:
                present = any(
                    warning_item == label and _point_in_box(*_box_center(b), (px1,py1,px2,py2))
                    for label, b, _ in all_other_dets
                )
                if not present:
                    missing_counts[warning_item] += 1
                    violator_flags[p_idx] = True

        violator_events += sum(1 for v in violator_flags if v)

        # --- Annotate Frame ---
        # The frame is already BGR here, so no conversion needed.
        annotated_bgr = frame_bgr 
        
        # Draw PPE
        for label, b, confidence in all_other_dets:
            color = (0,255,0)
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(annotated_bgr,(x1,y1),(x2,y2),color,2)
            cv2.putText(annotated_bgr,f"{label} {confidence:.2f}",(x1,max(16,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.45,color,1)
        
        # Draw persons
        for p_idx, (p_box, confidence) in enumerate(persons):
            x1, y1, x2, y2 = map(int, p_box)
            compliant = not violator_flags[p_idx]
            color = (0,255,0) if compliant else (0,0,255)
            cv2.rectangle(annotated_bgr,(x1,y1),(x2,y2),color,3)
            status = "COMPLIANT" if compliant else "VIOLATOR"
            cv2.putText(annotated_bgr,f"Person {status} {confidence:.2f}", (x1,max(16,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
        
        out.write(annotated_bgr)

    cap.release()
    out.release()
    
    return output_video_path, missing_counts, violator_events, persons_seen, detection_summary