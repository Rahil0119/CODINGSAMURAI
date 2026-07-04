import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from collections import deque, Counter
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# ✅ Two color formats: BGR for OpenCV, CSS string for Streamlit
dustbin_mapping = {
    #  label      : (bin name,      BGR tuple for cv2,  CSS string for HTML)
    "metal":     ("Yellow Bin", (0, 215, 255), "rgb(0, 215, 255)"),
    "glass":     ("Green Bin",  (0, 220, 80),  "rgb(0, 220, 80)"),
    "paper":     ("Blue Bin",   (255, 120, 0), "rgb(255, 120, 0)"),
    "cardboard": ("Blue Bin",   (255, 120, 0), "rgb(255, 120, 0)"),
    "plastic":   ("Yellow Bin", (0, 215, 255), "rgb(0, 215, 255)"),
    "trash":     ("Black Bin",  (160,160,160), "rgb(160, 160, 160)"),
}

# ✅ FIXED: Matches training exactly — no Normalize
transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.network = models.resnet50(weights=None)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs, num_classes)
        )
    def forward(self, xb):
        return self.network(xb)

def init_model(weights_path='garmodel-with-acc-93.pth'):
    model = ResNet(len(classes)).to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def predict_pil_image(image, model):
    img_t = transformations(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out  = model(img_t)
        prob = torch.nn.functional.softmax(out, dim=1)
        conf, idx = torch.max(prob, dim=1)
    return classes[idx.item()], conf.item() * 100, prob.cpu().numpy().flatten()

def predict_bgr_image(frame_bgr, model):
    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    return predict_pil_image(image, model)

# Object cropper and all other OpenCV helpers below unchanged...
class ObjectCropper:
    def __init__(self, min_area=2500, padding=20, stability_frames=2):
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=150, varThreshold=25, detectShadows=False
        )
        self.min_area         = min_area
        self.padding          = padding
        self.stable_count     = 0
        self.stability_frames = stability_frames
        self.last_box         = None
        self.prev_gray        = None
        self.BOX_DRIFT        = 80

    def _boxes_similar(self, b1, b2):
        return all(abs(b1[i] - b2[i]) < self.BOX_DRIFT for i in range(4))

    def detect(self, frame):
        h, w = frame.shape[:2]
        fg_mask = self.bg_sub.apply(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is not None:
            diff = cv2.absdiff(gray, self.prev_gray)
            _, diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            fg_mask = cv2.bitwise_or(fg_mask, diff)
        self.prev_gray = gray
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel, iterations=2)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.stable_count = 0
            return None, None, fg_mask
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.min_area:
            self.stable_count = 0
            return None, None, fg_mask
        bx, by, bw, bh = cv2.boundingRect(largest)
        x1 = max(0,   bx - self.padding)
        y1 = max(0,   by - self.padding)
        x2 = min(w-1, bx + bw + self.padding)
        y2 = min(h-1, by + bh + self.padding)
        box = (x1, y1, x2, y2)
        if self.last_box and self._boxes_similar(self.last_box, box):
            self.stable_count += 1
        else:
            self.stable_count = 1
        self.last_box = box
        return frame[y1:y2, x1:x2], box, fg_mask

ENTROPY_THRESHOLD = 1.0
CONF_THRESHOLD    = 55.0
vote_buffer       = deque(maxlen=15)
conf_buffer       = deque(maxlen=15)

def draw_corner_box(frame, x1, y1, x2, y2, color, length=30, thick=3):
    for (px, py, dx, dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (px,py), (px+dx*length, py),        color, thick)
        cv2.line(frame, (px,py), (px,           py+dy*length), color, thick)

def draw_pill(frame, text, x, y, fscale=0.7, fg=(255,255,255), bg=(25,25,25)):
    (tw,th),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fscale, 2)
    p = 7
    cv2.rectangle(frame, (x-p, y-th-p), (x+tw+p, y+p), bg, -1)
    cv2.putText(frame, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, fscale, fg, 2)

def draw_conf_bar(frame, conf, x, y, width=260, height=16):
    col = (0,200,80) if conf>=70 else (0,165,255) if conf>=50 else (60,60,200)
    cv2.rectangle(frame, (x,y), (x+width, y+height), (50,50,50), -1)
    cv2.rectangle(frame, (x,y), (x+int(width*conf/100), y+height), col, -1)
    cv2.putText(frame, f"{conf:.1f}%", (x+width+6, y+13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

def draw_mini_crop(frame, crop_bgr, x, y, size=100):
    if crop_bgr is None or crop_bgr.size == 0:
        return
    thumb = cv2.resize(crop_bgr, (size, size))
    frame[y:y+size, x:x+size] = thumb
    cv2.rectangle(frame, (x,y), (x+size, y+size), (200,200,200), 1)
    cv2.putText(frame, "Model sees:", (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160,160,160), 1)

def main():
    model = init_model()
    print("✅ Model loaded")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cropper     = ObjectCropper()
    frame_count = 0
    FRAME_SKIP  = 2
    print("📷 Show a garbage item | R=Reset | Q=Quit")

    while True:
        ret, frame = cap.read()
        if not ret: break
        h, w = frame.shape[:2]
        crop, box, _ = cropper.detect(frame)

        if crop is not None and frame_count % FRAME_SKIP == 0:
            img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            prediction, confidence, _ = predict_pil_image(img, model)
            if confidence >= CONF_THRESHOLD:
                vote_buffer.append(prediction)
                conf_buffer.append(confidence)

        frame_count += 1

        if box is None:
            cv2.putText(frame, "Place item in front of camera",
                        (w//2-220, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80,80,80), 2)
        else:
            x1,y1,x2,y2 = box
            if len(vote_buffer) == 0:
                draw_corner_box(frame, x1, y1, x2, y2, (0,165,255))
                draw_pill(frame, "Scanning...", x1, y1-12, fscale=0.6, fg=(0,165,255))
            else:
                smoothed_pred = Counter(vote_buffer).most_common(1)[0][0]
                smoothed_conf = sum(conf_buffer) / len(conf_buffer)
                # ✅ Unpack 3-tuple: name, bgr, css
                bin_name, bgr_color, _ = dustbin_mapping[smoothed_pred]
                draw_corner_box(frame, x1, y1, x2, y2, bgr_color, length=40, thick=3)
                draw_pill(frame, f"  {smoothed_pred.upper()}  ", x1, y1-14,
                          fscale=1.0, fg=bgr_color)
                cv2.putText(frame, "CONFIDENCE", (20,35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140,140,140), 1)
                draw_conf_bar(frame, smoothed_conf, 20, 42)
                btxt = f"  Throw in: {bin_name}  "
                (bw_,bh_),_ = cv2.getTextSize(btxt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                bx = (w-bw_)//2
                cv2.rectangle(frame,(bx-10,h-68),(bx+bw_+10,h-20),(20,20,20),-1)
                cv2.putText(frame, btxt, (bx, h-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, bgr_color, 2)
                draw_mini_crop(frame, crop, w-120, 10)

        cv2.putText(frame, "R=Reset  Q=Quit", (10, h-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (70,70,70), 1)
        cv2.imshow('Garbage Classifier', frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')): break
        if key == ord('r'):
            cropper = ObjectCropper()
            vote_buffer.clear(); conf_buffer.clear()
            print("🔄 Reset")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()