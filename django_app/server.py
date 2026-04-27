import os
import tempfile
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

# ----------------- Config -----------------
MODEL_PATH = "model_97_acc_60_frames_FF_data.pt"
DEVICE = torch.device("cpu")   # FORCE CPU (you said no CUDA driver)
ALLOWED_EXT = {'.mp4', '.mov', '.webm', '.mkv'}
MAX_FILE_MB = 600
NUM_FRAMES = 60
TARGET_SIZE = (112, 112)   # your model used 112x112 in the test snippet
# label mapping: index 0 -> real, index 1 -> fake
LABELS = {0: "real", 1: "fake"}
# ------------------------------------------

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# ----------------- Your corrected Model class -----------------
from torch import nn
from torchvision import models

class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        # Use resnext50 backbone and keep children up to the last conv block (same as training)
        backbone = models.resnext50_32x4d(pretrained=False)  # don't try to download weights on server
        # keep everything except final avgpool & fc -> children()[:-2]
        self.model = nn.Sequential(*list(backbone.children())[:-2])
        # LSTM: use batch_first=True so forward can accept (B, T, feat)
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers=lstm_layers, batch_first=True, bidirectional=bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # x: (B, T, C, H, W)
        batch_size, seq_length, c, h, w = x.shape
        # reshape to (B*T, C, H, W) to pass through backbone
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)             # e.g. (B*T, feat_dim, Hf, Wf)
        x = self.avgpool(fmap)           # (B*T, feat_dim, 1, 1)
        x = x.view(batch_size, seq_length, -1)  # (B, T, feat_dim)
        # LSTM expects (B, T, feat) with batch_first=True
        x_lstm, _ = self.lstm(x)         # (B, T, hidden)  (hidden doubled if bidirectional)
        # pool across time and pass through head
        logits = self.dp(self.linear1(torch.mean(x_lstm, dim=1)))  # (B, num_classes)
        return fmap, logits

# ----------------- Load model -----------------
model = None
def load_model(path):
    global model
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")

    # instantiate architecture exactly as used in training
    # adjust hidden_dim / lstm_layers / bidirectional if they differ from your training config
    net = Model(num_classes=2, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False)
    # load state_dict mapping to CPU
    state = torch.load(path, map_location=DEVICE)

    # state might be a dict with extra keys (if saved with prefix) or saved directly as state_dict
    if isinstance(state, dict) and ("state_dict" in state):
        # some training frameworks wrap state_dict
        state_dict = state["state_dict"]
    else:
        state_dict = state

    try:
        # load with strict True to catch missing keys; if you still get missing-key errors,
        # try strict=False and inspect warnings.
        net.load_state_dict(state_dict)
        print("Loaded state_dict with strict=True")
    except Exception as e:
        print("Strict load failed, trying strict=False. Error:", e)
        net.load_state_dict(state_dict, strict=False)
        print("Loaded state_dict with strict=False (some keys may be skipped)")

    net.to(DEVICE)
    net.eval()
    model = net
    print("Model loaded and ready on", DEVICE)

try:
    load_model(MODEL_PATH)
except Exception as e:
    print("Warning: model load failed:", e)
    model = None

# ----------------- Helpers -----------------
def allowed_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXT

def extract_frames(video_path, num_frames=NUM_FRAMES, target_size=TARGET_SIZE):
    """Extract `num_frames` uniformly from video and return (T, H, W, C) RGB uint8"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video file")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
        cap.release()
        total = len(frames)
        if total == 0:
            raise RuntimeError("No frames in video")
        idxs = np.linspace(0, total-1, num_frames, dtype=int)
        sampled = [frames[i] if i < len(frames) else frames[-1] for i in idxs]
    else:
        idxs = np.linspace(0, max(0, total-1), num_frames, dtype=int)
        sampled = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ret, frame = cap.read()
            if not ret:
                if sampled:
                    sampled.append(sampled[-1])
                else:
                    sampled.append(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))
                continue
            sampled.append(frame)
        cap.release()

    # Convert BGR->RGB and resize
    out = []
    for f in sampled:
        if f is None:
            f = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        try:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        except:
            f = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        f = cv2.resize(f, target_size, interpolation=cv2.INTER_AREA)
        out.append(f)
    arr = np.stack(out, axis=0)  # (T, H, W, C)
    return arr

def preprocess(frames_np):
    """
    Convert (T,H,W,C) uint8 -> torch tensor (1, C, T, H, W) float on DEVICE
    Matches architecture: frames normalized to [0,1]. If you trained with mean/std normalization,
    add it here (mean/std per channel).
    """
    frames = frames_np.astype(np.float32) / 255.0
    # transpose -> (C, T, H, W)
    frames = frames.transpose(3, 0, 1, 2)
    tensor = torch.from_numpy(frames).unsqueeze(0).to(DEVICE)  # (1, C, T, H, W)
    # model expects (B, T, C, H, W) in our forward; convert order below
    # change to (B, T, C, H, W)
    tensor = tensor.permute(0, 2, 1, 3, 4)  # (1, T, C, H, W)
    return tensor

def infer(tensor):
    if model is None:
        raise RuntimeError("Model not loaded on server.")
    model.eval()
    with torch.no_grad():
        # model.forward returns (fmap, logits)
        out = model(tensor)   # out is (fmap, logits)
        if isinstance(out, tuple) or isinstance(out, list):
            logits = out[1]
        else:
            logits = out
        logits = logits.cpu()
        # handle logits shape variations
        if logits.dim() == 2 and logits.size(1) == 2:
            probs = F.softmax(logits, dim=1).squeeze(0).numpy()
            prob_fake = float(probs[1])
        elif logits.dim() == 1 and logits.size(0) == 2:
            probs = F.softmax(logits, dim=0).numpy()
            prob_fake = float(probs[1])
        elif logits.dim() == 2 and logits.size(1) == 1:
            prob_fake = float(torch.sigmoid(logits).item())
        else:
            # fallback
            try:
                prob_fake = float(torch.sigmoid(logits.view(-1)[0]).item())
            except Exception:
                raise RuntimeError("Unrecognized model output shape.")
    return prob_fake

# ----------------- Routes -----------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    if 'video' not in request.files:
        return jsonify({"error": "No video provided"}), 400
    f = request.files['video']
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    # size check
    f.seek(0, os.SEEK_END)
    size_mb = f.tell() / (1024*1024)
    f.seek(0)
    if size_mb > MAX_FILE_MB:
        return jsonify({"error": f"File too large ({size_mb:.1f} MB). Max {MAX_FILE_MB} MB."}), 400

    tmp_path = None
    try:
        # Create a temp file name in a safe way for Windows (avoid open handle)
        suffix = os.path.splitext(f.filename)[1]
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)  # close the low-level descriptor so Flask can write the file
        # Save the uploaded file to the temp path
        f.save(tmp_path)

        # Now extract frames and run inference
        frames = extract_frames(tmp_path, num_frames=NUM_FRAMES, target_size=TARGET_SIZE)
        inp = preprocess(frames)  # (1, T, C, H, W)
        prob_fake = infer(inp)
        prediction = "fake" if prob_fake >= 0.5 else "real"
        confidence = float(prob_fake if prediction == "fake" else 1.0 - prob_fake)

        return jsonify({"prediction": prediction, "confidence": confidence})

    except Exception as e:
        # Log exception to console for debugging
        print("Predict endpoint error:", repr(e))
        return jsonify({"error": str(e)}), 500

    finally:
        # Always try to remove the temp file if it exists
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception as e_rm:
            print("Warning: failed to remove temp file:", tmp_path, repr(e_rm))


@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == "__main__":
    print("Starting server on CPU. Model loaded:", model is not None)
    app.run(host="0.0.0.0", port=7860, debug=True)
