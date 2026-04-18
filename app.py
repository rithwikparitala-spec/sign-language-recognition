import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from collections import deque, Counter
from mediapipe.python.solutions.holistic import Holistic, HAND_CONNECTIONS, POSE_CONNECTIONS
from mediapipe.python.solutions import drawing_utils as mp_drawing

st.set_page_config(page_title="Sign Language Recognition", page_icon="🤟", layout="wide")

st.markdown("""
<style>
    .sentence-box {
        background-color: #1e1e2e;
        border: 2px solid #00ff88;
        border-radius: 12px;
        padding: 20px 28px;
        font-size: 32px;
        font-weight: bold;
        color: #00ff88;
        min-height: 80px;
        letter-spacing: 2px;
        word-spacing: 8px;
    }
    .current-sign {
        background-color: #2e1e1e;
        border: 2px solid #ffaa00;
        border-radius: 12px;
        padding: 14px 24px;
        font-size: 24px;
        font-weight: bold;
        color: #ffaa00;
        text-align: center;
    }
    .status-box {
        background-color: #1e1e2e;
        border-radius: 8px;
        padding: 10px 18px;
        font-size: 16px;
        color: #aaaaaa;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ─── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH    = r"C:\Users\parit\sign_language_kp_model.h5"
LABELS_PATH   = r"C:\Users\parit\lsa64_labels.npy"
WORDS_PATH    = r"C:\Users\parit\labels.txt"
SEQ_LENGTH    = 30
THRESHOLD     = 0.70
SMOOTH_N      = 7
NO_HAND_LIMIT = 20
PREDICT_EVERY = 5

# ─── LOAD WORD MAP from labels.txt ─────────────────────────────────────────────
@st.cache_resource
def load_word_map(words_path):
    """
    Parse labels.txt into a dict: {'001': 'Opaque', '002': 'Red', ...}
    Handles both zero-padded (001) and non-padded (1) numbers.
    """
    word_map = {}
    with open(words_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                num  = parts[0].strip().zfill(3)  # pad to 3 digits: "01" → "001"
                word = parts[1].strip()
                word_map[num] = word
    print("Word map loaded:", word_map)
    return word_map

# ─── LOAD MODEL ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_labels():
    model  = tf.keras.models.load_model(MODEL_PATH)
    labels = np.load(LABELS_PATH, allow_pickle=True).tolist()
    model.predict(np.zeros((1, SEQ_LENGTH, 1662), dtype=np.float32), verbose=0)
    print("Model loaded. Labels:", labels[:5], "...")
    return model, labels

model, labels   = load_model_and_labels()
word_map        = load_word_map(WORDS_PATH)

def get_word(label):
    """Convert label like '051' to 'Thanks'."""
    return word_map.get(label, label)  # fallback to label number if not found

# ─── SESSION STATE ─────────────────────────────────────────────────────────────
if "sentence" not in st.session_state: st.session_state.sentence = []
if "run"      not in st.session_state: st.session_state.run      = False

# ─── KEYPOINT EXTRACTION ───────────────────────────────────────────────────────
def extract_keypoints(results):
    pose = np.array(
        [[l.x, l.y, l.z, l.visibility] for l in results.pose_landmarks.landmark]
        if results.pose_landmarks else np.zeros((33, 4))
    ).flatten()
    face = np.array(
        [[l.x, l.y, l.z] for l in results.face_landmarks.landmark]
        if results.face_landmarks else np.zeros((468, 3))
    ).flatten()
    lh = np.array(
        [[l.x, l.y, l.z] for l in results.left_hand_landmarks.landmark]
        if results.left_hand_landmarks else np.zeros((21, 3))
    ).flatten()
    rh = np.array(
        [[l.x, l.y, l.z] for l in results.right_hand_landmarks.landmark]
        if results.right_hand_landmarks else np.zeros((21, 3))
    ).flatten()
    return np.concatenate([pose, face, lh, rh])

# ─── UI ────────────────────────────────────────────────────────────────────────
st.title("🤟 Sign Language to Sentence")
st.markdown("Show a sign → hold it → lower your hand → word gets added to the sentence.")

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("📷 Camera Feed")
    frame_window = st.image([])

with col2:
    st.subheader("🟡 Current Sign")
    current_sign_box = st.empty()
    st.subheader("📊 Status")
    status_box = st.empty()
    st.subheader("📈 Confidence")
    conf_bar = st.empty()
    st.subheader("🖐 Hand Detection")
    hand_status_box = st.empty()

st.subheader("📝 Sentence")
sentence_box = st.empty()

c1, c2, c3 = st.columns(3)
with c1: start = st.button("▶ Start", use_container_width=True)
with c2: clear = st.button("🗑 Clear", use_container_width=True)
with c3: stop  = st.button("⏹ Stop",  use_container_width=True)

if clear:
    st.session_state.sentence = []
if stop:
    st.session_state.run = False
if start:
    st.session_state.run = True

# ─── MAIN LOOP ─────────────────────────────────────────────────────────────────
if st.session_state.run:

    frame_buffer  = deque(maxlen=SEQ_LENGTH)
    pred_buffer   = deque(maxlen=SMOOTH_N)
    no_hand_count = 0
    sign_locked   = False
    locked_label  = ""   # raw label like "051"
    locked_word   = ""   # mapped word like "Thanks"
    locked_conf   = 0.0
    frame_count   = 0

    cap = cv2.VideoCapture(0)

    with Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as holistic:

        while cap.isOpened() and st.session_state.run:

            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img.flags.writeable = False
            results = holistic.process(img)
            img.flags.writeable = True

            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(img, results.left_hand_landmarks, HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(img, results.right_hand_landmarks, HAND_CONNECTIONS)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(img, results.pose_landmarks, POSE_CONNECTIONS)

            hand_detected = bool(
                results.left_hand_landmarks or results.right_hand_landmarks
            )

            if not hand_detected:
                no_hand_count += 1
            else:
                no_hand_count = 0

            # sign complete when hand gone for NO_HAND_LIMIT frames
            if sign_locked and no_hand_count >= NO_HAND_LIMIT:
                if locked_word and (
                    not st.session_state.sentence or
                    st.session_state.sentence[-1] != locked_word
                ):
                    st.session_state.sentence.append(locked_word)
                sign_locked  = False
                locked_label = ""
                locked_word  = ""
                locked_conf  = 0.0
                frame_buffer.clear()
                pred_buffer.clear()

            kp = extract_keypoints(results)
            frame_buffer.append(kp)

            # predict every PREDICT_EVERY frames when hand present
            if (hand_detected and
                len(frame_buffer) == SEQ_LENGTH and
                frame_count % PREDICT_EVERY == 0):

                seq   = np.expand_dims(np.array(frame_buffer), axis=0)
                probs = model.predict(seq, verbose=0)[0]
                pred_idx   = int(np.argmax(probs))
                pred_conf  = float(probs[pred_idx])
                pred_label = labels[pred_idx]
                pred_word  = get_word(pred_label)

                print(f"Pred: {pred_label} ({pred_word}) | Conf: {pred_conf:.2f}")

                if pred_conf >= THRESHOLD:
                    pred_buffer.append(pred_idx)
                else:
                    pred_buffer.clear()

                if len(pred_buffer) == SMOOTH_N:
                    majority, count = Counter(pred_buffer).most_common(1)[0]
                    if majority != -1 and count >= 5:
                        locked_label = labels[majority]
                        locked_word  = get_word(locked_label)
                        locked_conf  = pred_conf
                        sign_locked  = True
                        print(f"LOCKED: {locked_label} → {locked_word} ({locked_conf:.2f})")

            # ── UI ────────────────────────────────────────────────────────
            frame_window.image(img, channels="RGB", use_container_width=True)

            # hand indicator
            if hand_detected:
                hand_status_box.markdown(
                    '<div style="background:#1a3a1a;border:2px solid #00ff88;border-radius:8px;'
                    'padding:8px;text-align:center;color:#00ff88;font-weight:bold;">✅ Hand Detected</div>',
                    unsafe_allow_html=True
                )
            else:
                hand_status_box.markdown(
                    '<div style="background:#3a1a1a;border:2px solid #ff4444;border-radius:8px;'
                    'padding:8px;text-align:center;color:#ff4444;font-weight:bold;">❌ No Hand</div>',
                    unsafe_allow_html=True
                )

            # current sign box — shows word not number
            if sign_locked and locked_word:
                current_sign_box.markdown(
                    f'<div class="current-sign">🤟 {locked_word}<br>'
                    f'<span style="font-size:14px;color:#aaa">{locked_label} — {locked_conf*100:.0f}% confidence</span></div>',
                    unsafe_allow_html=True
                )
            else:
                current_sign_box.markdown(
                    '<div class="current-sign" style="color:#666">Waiting for sign...</div>',
                    unsafe_allow_html=True
                )

            if no_hand_count >= NO_HAND_LIMIT:
                status_text = "✋ Lower hand to confirm sign"
            elif sign_locked:
                status_text = f"🔒 Locked: {locked_word}"
            elif hand_detected:
                status_text = f"👁 Detecting... ({len(frame_buffer)}/{SEQ_LENGTH})"
            else:
                status_text = "⏳ Waiting..."

            status_box.markdown(
                f'<div class="status-box">{status_text}</div>',
                unsafe_allow_html=True
            )

            conf_bar.progress(int(locked_conf * 100) if sign_locked else 0)

            # sentence shows actual words
            sentence_text = " ".join(st.session_state.sentence) if st.session_state.sentence else "..."
            sentence_box.markdown(
                f'<div class="sentence-box">{sentence_text}</div>',
                unsafe_allow_html=True
            )

    cap.release()
    st.session_state.run = False
    st.success("Stopped.")