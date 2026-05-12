"""
WasteNet — Local Inference Server
Run: python app.py
Then open waste_nn_dashboard.html in your browser.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import io, time, os

app = Flask(__name__)
CORS(app)

CLASSES = ['plastic', 'paper_cardboard', 'metal', 'glass', 'organic', 'e_waste']
DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT    = os.path.join(os.path.dirname(__file__), 'checkpoints', 'vit_full_best.pth')

# ── Attention modules ────────────────────────────────────────────────────────

class ViTModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            'vit_small_patch16_224',
            pretrained=False,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.backbone(x)


# ── Load model ───────────────────────────────────────────────────────────────

print(f"Loading model from: {CKPT}")
print(f"Device: {DEVICE}")

if not os.path.exists(CKPT):
    print(f"\n⚠  Checkpoint not found at {CKPT}")
    print("   Place vit_full_best.pth inside the checkpoints/ folder.\n")
    model = None
else:
    model = ViTModel().to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()
    print("✅ Model loaded successfully\n")

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status' : 'ok',
        'device' : str(DEVICE),
        'model'  : 'loaded' if model else 'missing checkpoint',
        'classes': CLASSES,
    })


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model checkpoint not found. See checkpoints/README.txt'}), 503

    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No image file in request'}), 400

    try:
        img    = Image.open(io.BytesIO(file.read())).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(DEVICE)
    except Exception as e:
        return jsonify({'error': f'Could not read image: {e}'}), 400

    t0 = time.time()
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().tolist()
    latency_ms = round((time.time() - t0) * 1000, 1)

    result  = {cls: round(p, 4) for cls, p in zip(CLASSES, probs)}
    top_cls = max(result, key=result.get)

    return jsonify({
        'predicted'    : top_cls,
        'confidence'   : result[top_cls],
        'probabilities': result,
        'latency_ms'   : latency_ms,
    })


if __name__ == '__main__':
    print("🚀 Starting WasteNet server at http://127.0.0.1:5000")
    print("   Open waste_nn_dashboard.html in your browser.\n")
    app.run(host='127.0.0.1', port=5000, debug=False)

