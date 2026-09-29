"""OnionIQ local inference server. Serves the UI and runs best.pt (YOLO: Defective/Healthy/Sprouted/Undersized)."""
import io, os
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("MODEL", os.path.join(HERE, "model", "best.pt"))
IMGSZ = int(os.environ.get("IMGSZ", 640)); CONF = float(os.environ.get("CONF", 0.25))
KEY = {"Healthy": "h", "Sprouted": "s", "Defective": "r", "Undersized": "u"}  # UI class keys
model = YOLO(MODEL)
app = Flask(__name__, static_folder=os.path.join(HERE, "static"), static_url_path="")

@app.route("/")
def index(): return app.send_static_file("index.html")

def predict(imgs): return model.predict(imgs, imgsz=IMGSZ, conf=CONF, max_det=300, verbose=False)
def load(f): return Image.open(io.BytesIO(f.read())).convert("RGB")

@app.post("/api/analyze")
def analyze():
    try: im = load(request.files["image"])
    except Exception as e: return jsonify(error="Bad image: %s" % e), 400
    r = predict([np.array(im)])[0]
    out = [dict(k=KEY.get(model.names[int(c)], "r"), name=model.names[int(c)], conf=float(p), box=[round(v, 4) for v in b])
           for c, p, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxyn.tolist())]
    return jsonify(detections=out, width=im.width, height=im.height)

@app.post("/api/explain")
def explain():
    """Occlusion sensitivity on the real model: grey out each cell of the bulb, measure confidence drop."""
    im = np.array(load(request.files["image"])); H, W = im.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in request.form["box"].split(",")]; name = request.form["name"]; N = 5
    X1, Y1, X2, Y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)
    def score(res):
        best = 0.0
        for c, p, b in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist(), res.boxes.xyxyn.tolist()):
            if model.names[int(c)] != name: continue
            iw = min(b[2], x2) - max(b[0], x1); ih = min(b[3], y2) - max(b[1], y1)
            inter = max(iw, 0) * max(ih, 0); u = (b[2]-b[0])*(b[3]-b[1]) + (x2-x1)*(y2-y1) - inter
            if u > 0 and inter / u > .4: best = max(best, p)
        return best
    base = score(predict([im])[0]); variants = []
    for gy in range(N):
        for gx in range(N):
            v = im.copy(); v[Y1 + (Y2-Y1)*gy//N: Y1 + (Y2-Y1)*(gy+1)//N, X1 + (X2-X1)*gx//N: X1 + (X2-X1)*(gx+1)//N] = 114
            variants.append(v)
    drops = [max(0.0, base - score(r)) for r in predict(variants)]
    m = max(drops) or 1.0
    return jsonify(grid=[d / m for d in drops], n=N, base=base)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
