from flask import Flask, Response, render_template_string
import cv2
import numpy as np
import tensorflow as tf
from collections import deque, Counter
from mediapipe.python.solutions.holistic import Holistic, HAND_CONNECTIONS, POSE_CONNECTIONS
from mediapipe.python.solutions import drawing_utils as mp_drawing
import threading
import time

# ─── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH  = r"C:\Users\parit\sign_language_kp_model.h5"
LABELS_PATH = r"C:\Users\parit\lsa64_labels.npy"
WORDS_PATH  = r"C:\Users\parit\labels.txt"
SEQ_LENGTH  = 30
THRESHOLD   = 0.80
SMOOTH_N    = 7
NO_HAND_LIMIT = 20

# ─── LOAD RESOURCES ────────────────────────────────────────────────────────────
print("Loading model...")
model  = tf.keras.models.load_model(MODEL_PATH)
labels = np.load(LABELS_PATH, allow_pickle=True).tolist()
model.predict(np.zeros((1, SEQ_LENGTH, 1662), dtype=np.float32), verbose=0)
print("Model ready.")

# load word map
word_map = {}
with open(WORDS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        parts = line.split("\t")
        if len(parts) >= 2:
            word_map[parts[0].strip().zfill(3)] = parts[1].strip()

def get_word(label):
    return word_map.get(label, label)

# ─── SHARED STATE ──────────────────────────────────────────────────────────────
state = {
    "current_word" : "",
    "current_conf" : 0.0,
    "sentence"     : [],
    "status"       : "Waiting...",
    "hand"         : False,
    "buf_len"      : 0,
    "locked"       : False,
}
state_lock = threading.Lock()

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

# ─── CAMERA + PREDICTION THREAD ───────────────────────────────────────────────
output_frame = None
frame_lock   = threading.Lock()
running      = True

def camera_thread():
    global output_frame, running

    frame_buffer  = deque(maxlen=SEQ_LENGTH)
    pred_buffer   = deque(maxlen=SMOOTH_N)
    no_hand_count = 0
    sign_locked   = False
    locked_label  = ""
    locked_word   = ""
    locked_conf   = 0.0

    cap = cv2.VideoCapture(0)

    with Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as holistic:

        while running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img.flags.writeable = False
            results = holistic.process(img)
            img.flags.writeable = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

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

            # sign complete when hand gone
            if sign_locked and no_hand_count >= NO_HAND_LIMIT:
                with state_lock:
                    if locked_word and (
                        not state["sentence"] or
                        state["sentence"][-1] != locked_word
                    ):
                        state["sentence"].append(locked_word)
                sign_locked  = False
                locked_label = ""
                locked_word  = ""
                locked_conf  = 0.0
                frame_buffer.clear()
                pred_buffer.clear()

            # only collect keypoints when hand present
            if hand_detected:
                kp = extract_keypoints(results)
                frame_buffer.append(kp)
            else:
                # clear buffer when no hand — prevents stale frames mixing in
                if no_hand_count > 5:
                    frame_buffer.clear()
                    pred_buffer.clear()

            # predict when buffer full and hand present
            if hand_detected and len(frame_buffer) == SEQ_LENGTH:
                seq   = np.expand_dims(np.array(frame_buffer), axis=0)
                probs = model.predict(seq, verbose=0)[0]
                pred_idx  = int(np.argmax(probs))
                pred_conf = float(probs[pred_idx])
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
                        print(f"LOCKED: {locked_label} → {locked_word}")

            # draw overlay on frame
            h, w = img.shape[:2]

            # top banner
            cv2.rectangle(img, (0, 0), (w, 70), (20, 20, 40), -1)

            if sign_locked and locked_word:
                cv2.putText(img, f"{locked_word}  ({locked_conf*100:.0f}%)",
                            (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                            (0, 255, 150), 2, cv2.LINE_AA)
            else:
                cv2.putText(img, "Waiting for sign...",
                            (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (150, 150, 150), 2, cv2.LINE_AA)

            # buffer progress bar
            buf_w = int((len(frame_buffer) / SEQ_LENGTH) * w)
            cv2.rectangle(img, (0, h - 10), (buf_w, h), (0, 180, 255), -1)

            # hand indicator dot
            dot_color = (0, 255, 0) if hand_detected else (0, 0, 220)
            cv2.circle(img, (w - 30, 35), 12, dot_color, -1)

            # sentence at bottom
            with state_lock:
                sentence = state["sentence"][:]
            sentence_text = " ".join(sentence)
            if sentence_text:
                cv2.rectangle(img, (0, h - 50), (w, h - 10), (20, 20, 40), -1)
                cv2.putText(img, sentence_text,
                            (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 100), 2, cv2.LINE_AA)

            # update state
            with state_lock:
                state["current_word"] = locked_word if sign_locked else ""
                state["current_conf"] = locked_conf
                state["hand"]         = hand_detected
                state["locked"]       = sign_locked
                state["buf_len"]      = len(frame_buffer)

            # encode frame
            _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            with frame_lock:
                output_frame = buffer.tobytes()

    cap.release()

# ─── FLASK APP ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Sign Language Recognition</title>
    <meta charset="UTF-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #0f0f1a; color: #fff; font-family: Arial, sans-serif; }
        .container { display: flex; gap: 20px; padding: 20px; }
        .video-col { flex: 2; }
        .info-col { flex: 1; display: flex; flex-direction: column; gap: 16px; }
        h1 { text-align: center; padding: 16px; font-size: 24px; color: #00ff88; border-bottom: 1px solid #333; }
        img { width: 100%; border-radius: 12px; border: 2px solid #333; }
        .card { background: #1e1e2e; border-radius: 12px; padding: 16px; }
        .card h3 { font-size: 13px; color: #888; margin-bottom: 8px; text-transform: uppercase; }
        .sign-display { font-size: 28px; font-weight: bold; color: #ffaa00; text-align: center; padding: 10px; }
        .conf { font-size: 13px; color: #888; text-align: center; }
        .sentence { font-size: 22px; font-weight: bold; color: #00ff88; min-height: 50px; padding: 8px; word-spacing: 6px; }
        .status { font-size: 14px; color: #aaa; text-align: center; }
        .hand-ok  { color: #00ff88; font-weight: bold; text-align: center; font-size: 15px; }
        .hand-no  { color: #ff4444; font-weight: bold; text-align: center; font-size: 15px; }
        button { width: 100%; padding: 12px; border: none; border-radius: 8px; font-size: 16px; font-weight: bold; cursor: pointer; margin-top: 4px; }
        .btn-clear { background: #ff4444; color: white; }
        .btn-clear:hover { background: #cc0000; }
        .progress-bar { background: #333; border-radius: 4px; height: 8px; margin-top: 6px; }
        .progress-fill { background: #0080ff; height: 8px; border-radius: 4px; transition: width 0.1s; }
    </style>
</head>
<body>
    <h1>🤟 Sign Language to Sentence</h1>
    <div class="container">
        <div class="video-col">
            <img src="/video" />
        </div>
        <div class="info-col">
            <div class="card">
                <h3>Current Sign</h3>
                <div class="sign-display" id="current-sign">Waiting...</div>
                <div class="conf" id="current-conf"></div>
            </div>
            <div class="card">
                <h3>Hand Detection</h3>
                <div id="hand-status" class="hand-no">❌ No Hand</div>
            </div>
            <div class="card">
                <h3>Buffer</h3>
                <div class="status" id="buf-status">0 / 30</div>
                <div class="progress-bar">
                    <div class="progress-fill" id="buf-bar" style="width:0%"></div>
                </div>
            </div>
            <div class="card">
                <h3>Sentence</h3>
                <div class="sentence" id="sentence">...</div>
            </div>
            <button class="btn-clear" onclick="clearSentence()">🗑 Clear Sentence</button>
        </div>
    </div>
    <script>
        function update() {
            fetch('/state')
                .then(r => r.json())
                .then(d => {
                    document.getElementById('current-sign').innerText = d.current_word || 'Waiting...';
                    document.getElementById('current-conf').innerText = d.current_word ? `${Math.round(d.current_conf * 100)}% confidence` : '';
                    document.getElementById('hand-status').innerText = d.hand ? '✅ Hand Detected' : '❌ No Hand';
                    document.getElementById('hand-status').className = d.hand ? 'hand-ok' : 'hand-no';
                    document.getElementById('sentence').innerText = d.sentence.length ? d.sentence.join(' ') : '...';
                    document.getElementById('buf-status').innerText = `${d.buf_len} / 30`;
                    document.getElementById('buf-bar').style.width = `${(d.buf_len / 30) * 100}%`;
                });
        }
        function clearSentence() {
            fetch('/clear').then(() => update());
        }
        setInterval(update, 300);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video')
def video():
    def generate():
        while True:
            with frame_lock:
                frame = output_frame
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/state')
def get_state():
    with state_lock:
        return {
            "current_word" : state["current_word"],
            "current_conf" : state["current_conf"],
            "sentence"     : state["sentence"],
            "hand"         : state["hand"],
            "buf_len"      : state["buf_len"],
            "locked"       : state["locked"],
        }

@app.route('/clear')
def clear():
    with state_lock:
        state["sentence"] = []
    return "OK"

# ─── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    t = threading.Thread(target=camera_thread, daemon=True)
    t.start()
    print("Open your browser at: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)