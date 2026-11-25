"""Stage 6: who is on screen, and is it the same person as before?

Finds people in frames sampled about once per second, follows each one
through a shot, then links appearances across the whole video into
characters:

- **Face identity (F1, F2 ...):** faces are matched with a face-embedding
  model, so the same person is recognised in other scenes and other clothes.
- **Outfit identity (O1, O2 ...):** clothing or suit appearance from a
  person re-identification model plus named colours. Links appearances where
  no face is visible (back to camera, helmet, mask, armour suit) to a known
  character, or groups them into a character of their own.
- **Character (C1, C2 ...):** one ID that ties a person's faces, outfits and,
  where it can be inferred, their voice group (S1, S2 ...) together. C1 has
  the most screen time.

Everything runs locally on the CPU with OpenCV and four small ONNX models
from the OpenCV model zoo (MIT / Apache 2.0, about 180 MB in total,
downloaded once). People are only grouped *within one video*; nobody is
named or looked up, and face embeddings are never written to the output.
"""
from __future__ import annotations

import hashlib
import math
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

SAMPLE_FPS = 1.0
FRAME_WIDTH = 960
MIN_PERSON_HEIGHT = 0.15      # share of frame height; smaller figures are background
MIN_FACE_PIXELS = 36          # smallest face side used for recognition
FACE_THRESHOLD = 0.40         # cosine similarity to a face group's centre
OUTFIT_THRESHOLD = 0.72       # outfit similarity to an outfit group's centre
TRACK_IOU = 0.25
TRACK_APPEARANCE = 0.70

ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
MODELS = {
    "person": ("object_detection_yolox/object_detection_yolox_2022nov.onnx",
               "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"),
    "face": ("face_detection_yunet/face_detection_yunet_2023mar.onnx",
             "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    "face_id": ("face_recognition_sface/face_recognition_sface_2021dec.onnx",
                "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
    "body_id": ("person_reid_youtureid/person_reid_youtu_2021nov.onnx",
                "0579683334d4b9440221606dcb461656dd0dc64143b18f48faedaced9b4f580d"),
}

COLOUR_NAMES = ["black", "white", "grey", "red", "orange", "yellow", "green",
                "teal", "blue", "purple", "pink", "brown"]


def _np():
    try:
        import numpy
    except ImportError as exc:
        raise RuntimeError("Character analysis requires NumPy: python -m pip install numpy") from exc
    return numpy


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Character analysis requires OpenCV: "
                           "python -m pip install opencv-python-headless") from exc
    return cv2


def _r(x, d=2):
    return None if x is None else round(float(x), d)


# --------------------------------------------------------------- models ----

def model_dir():
    root = os.environ.get("TRUE_EXPERIENCE_MODELS")
    return Path(root) if root else Path.home() / ".cache" / "true_experience" / "models"


def ensure_model(name, directory=None):
    """Return the local path of a zoo model, downloading and checking it once."""
    rel, digest = MODELS[name]
    directory = Path(directory) if directory else model_dir()
    path = directory / Path(rel).name
    if path.exists() and _sha256(path) == digest:
        return path
    directory.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    print(f"Downloading {path.name} (once) ...", file=sys.stderr)
    try:
        urllib.request.urlretrieve(ZOO + rel, tmp)
    except OSError as exc:
        raise RuntimeError(f"Could not download {path.name} from the OpenCV model zoo: {exc}. "
                           f"You can place the file in {directory} yourself.") from exc
    if _sha256(tmp) != digest:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch for {path.name}; download again")
    tmp.replace(path)
    return path


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------- colour ----

def colour_histogram(bgr):
    """Share of pixels in each of COLOUR_NAMES for an image region."""
    np, cv2 = _np(), _cv2()
    if bgr is None or bgr.size == 0:
        return np.zeros(len(COLOUR_NAMES))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float64)
    h, s, v = hsv[:, 0] * 2.0, hsv[:, 1] / 255.0, hsv[:, 2] / 255.0
    idx = np.full(len(h), 2)                      # grey by default
    chroma = (s >= 0.2) & (v >= 0.2)
    hue_bins = [(15, 3), (40, 4), (70, 5), (160, 6), (190, 7), (260, 8), (290, 9), (335, 10), (361, 3)]
    hue_idx = np.full(len(h), 3)
    for limit, colour in reversed(hue_bins):
        hue_idx = np.where(h < limit, colour, hue_idx)
    brown = (hue_idx == 4) & (v < 0.6) | (hue_idx == 3) & (v < 0.45) & (s < 0.7)
    hue_idx = np.where(brown, 11, hue_idx)
    idx = np.where(chroma, hue_idx, idx)
    idx = np.where(~chroma & (v > 0.8) & (s < 0.2), 1, idx)
    idx = np.where(v < 0.2, 0, idx)
    hist = np.bincount(idx, minlength=len(COLOUR_NAMES)).astype(np.float64)
    return hist / max(1.0, hist.sum())


def colour_words(hist, limit=2, min_share=0.2):
    order = sorted(range(len(hist)), key=lambda i: -hist[i])
    words = [COLOUR_NAMES[i] for i in order[:limit] if hist[i] >= min_share]
    return words or [COLOUR_NAMES[order[0]]]


# ------------------------------------------------------------- detection ----

class OpenCVPeople:
    """Person boxes (YOLOX), faces (YuNet), face embeddings (SFace) and
    outfit embeddings (Youtu ReID), all with OpenCV DNN on the CPU."""

    def __init__(self, directory=None, person_score=0.45, face_score=0.8):
        cv2 = _cv2()
        self.cv2, self.np = cv2, _np()
        self.person = cv2.dnn.readNet(str(ensure_model("person", directory)))
        self.face = cv2.FaceDetectorYN.create(str(ensure_model("face", directory)), "", (320, 320),
                                              face_score)
        self.face_id = cv2.FaceRecognizerSF.create(str(ensure_model("face_id", directory)), "")
        self.body_id = cv2.dnn.readNet(str(ensure_model("body_id", directory)))
        self.person_score = person_score
        grids, strides = [], []
        for s in (8, 16, 32):
            n = 640 // s
            xv, yv = self.np.meshgrid(self.np.arange(n), self.np.arange(n))
            grids.append(self.np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(self.np.full((n * n, 1), s))
        self.grids, self.strides = self.np.concatenate(grids), self.np.concatenate(strides)

    def persons(self, frame):
        np, cv2 = self.np, self.cv2
        h, w = frame.shape[:2]
        scale = min(640 / h, 640 / w)
        padded = np.full((640, 640, 3), 114, np.float32)
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (int(w * scale), int(h * scale)))
        padded[:small.shape[0], :small.shape[1]] = small
        self.person.setInput(np.transpose(padded, (2, 0, 1))[None])
        out = self.person.forward(self.person.getUnconnectedOutLayersNames())[0][0]
        score = out[:, 4] * out[:, 5]                     # COCO class 0 = person
        keep = score > self.person_score
        xy = (out[keep, :2] + self.grids[keep]) * self.strides[keep]
        wh = np.exp(out[keep, 2:4]) * self.strides[keep]
        boxes = np.concatenate([xy - wh / 2, wh], axis=1) / scale
        chosen = cv2.dnn.NMSBoxes(boxes.tolist(), score[keep].tolist(), self.person_score, 0.5)
        return [(_clip_box(boxes[i], w, h), float(score[keep][i])) for i in np.array(chosen).flatten()]

    def faces(self, frame):
        h, w = frame.shape[:2]
        self.face.setInputSize((w, h))
        _, rows = self.face.detect(frame)
        return [] if rows is None else [r for r in rows]

    def face_embedding(self, frame, row):
        feat = self.face_id.feature(self.face_id.alignCrop(frame, row)).flatten()
        return feat / max(1e-9, float(self.np.linalg.norm(feat)))

    def body_embedding(self, crop):
        np, cv2 = self.np, self.cv2
        img = cv2.resize(crop, (128, 256))[:, :, ::-1] / 255.0
        img = ((img - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)).astype(np.float32)
        self.body_id.setInput(cv2.dnn.blobFromImage(img))
        feat = self.body_id.forward().flatten()
        return feat / max(1e-9, float(np.linalg.norm(feat)))

    def __call__(self, frame):
        return describe_people(frame, self.persons(frame), self.faces(frame),
                               self.face_embedding, self.body_embedding)

    def close(self):
        self.person = self.face = self.face_id = self.body_id = None


def _clip_box(box, w, h):
    x, y, bw, bh = [float(v) for v in box]
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(float(w), x + bw), min(float(h), y + bh)
    return [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)]


def describe_people(frame, persons, faces, face_embedding, body_embedding):
    """Pair faces with person boxes and compute features for each person."""
    h, w = frame.shape[:2]
    people = [{"box": list(b), "score": s, "face": None} for b, s in persons]
    for row in faces:
        fx, fy, fw, fh = [float(v) for v in row[:4]]
        cx, cy = fx + fw / 2, fy + fh / 2
        best = None
        for p in people:
            x, y, bw, bh = p["box"]
            # Face centre in the upper part of the body box (close-ups fill the box).
            if p["face"] is None and x <= cx <= x + bw and y <= cy <= y + 0.75 * bh:
                if best is None or bw * bh < best["box"][2] * best["box"][3]:
                    best = p
        if best is None:  # face without a detected body: assume one below it
            best = {"box": _clip_box([fx - fw, fy - 0.3 * fh, 3 * fw, 6.5 * fh], w, h),
                    "score": float(row[-1]), "face": None, "box_from_face": True}
            people.append(best)
        best["face"] = {"box": [fx, fy, fw, fh], "score": float(row[-1]), "row": row}
    result = []
    for p in people:
        x, y, bw, bh = p["box"]
        if bh < MIN_PERSON_HEIGHT * h and p["face"] is None:
            continue
        crop = frame[int(y):int(y + bh), int(x):int(x + bw)]
        if crop.size == 0:
            continue
        torso = crop[int(0.2 * crop.shape[0]):int(0.6 * crop.shape[0]),
                     int(0.2 * crop.shape[1]):int(0.8 * crop.shape[1])]
        face = p["face"]
        face_feat = None
        if face and min(face["box"][2], face["box"][3]) >= MIN_FACE_PIXELS:
            face_feat = face_embedding(frame, face["row"])
        result.append({
            "box": [round(v, 1) for v in p["box"]], "score": round(p["score"], 3),
            "frame_share": bw * bh / float(w * h),
            "face_box": [round(v, 1) for v in face["box"]] if face else None,
            "face_embedding": face_feat,
            "body_embedding": body_embedding(crop) if crop.shape[0] >= 32 and crop.shape[1] >= 16 else None,
            "colours": colour_histogram(torso)})
    return result


# --------------------------------------------------------------- frames ----

def sample_frames(video, info, fps=SAMPLE_FPS, width=FRAME_WIDTH):
    """Yield (seconds, BGR frame) about ``fps`` times per second, from one FFmpeg decode."""
    np = _np()
    width, height = frame_size(info, width)
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-i", str(video), "-map", "0:v:0",
           "-vf", f"fps={fps},scale={width}:{height}", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    size = width * height * 3
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as proc:
        i = 0
        while True:
            data = proc.stdout.read(size)
            if len(data) < size:
                break
            yield (i + 0.5) / fps, np.frombuffer(data, np.uint8).reshape(height, width, 3)
            i += 1
        proc.stdout.close()
        proc.wait()


def frame_size(info, width=FRAME_WIDTH):
    """Size of sampled frames; boxes in the output use these pixel coordinates."""
    src_w, src_h = info.get("width") or 0, info.get("height") or 0
    if not src_w or not src_h:
        raise RuntimeError("Video width and height unknown; cannot sample frames")
    w = min(width, int(src_w)) // 2 * 2
    return w, max(2, round(w * src_h / src_w / 2) * 2)


def frame_at(video, info, t, width=FRAME_WIDTH):
    """One BGR frame at ``t`` seconds, at the same size as sampled frames."""
    np = _np()
    w, h = frame_size(info, width)
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-ss", f"{max(0.0, t):.3f}",
           "-i", str(video), "-map", "0:v:0", "-frames:v", "1", "-vf", f"scale={w}:{h}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    data = subprocess.run(cmd, capture_output=True).stdout
    if len(data) < w * h * 3:
        raise RuntimeError(f"Could not decode a frame at {t:.2f}s")
    return np.frombuffer(data[:w * h * 3], np.uint8).reshape(h, w, 3)


def shot_index(shots, t):
    for i, s in enumerate(shots):
        if s["start_seconds"] <= t < s["end_seconds"]:
            return i
    return len(shots) - 1 if shots else 0


# -------------------------------------------------------------- tracking ----

def _iou(a, b):
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax1, bx1) - max(a[0], b[0]))
    ih = max(0.0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _cos(a, b):
    if a is None or b is None:
        return None
    return float(_np().dot(a, b))


class Tracker:
    """Follows people from one sample to the next inside a shot."""

    def __init__(self, fps=SAMPLE_FPS):
        self.fps = fps
        self.tracks = []
        self.active = []
        self.shot = None
        self.frame_index = -1

    def update(self, t, shot, people):
        self.frame_index += 1
        if shot != self.shot:
            self.active, self.shot = [], shot
        pairs = []
        for i, tr in enumerate(self.active):
            last = tr["detections"][-1]
            for j, p in enumerate(people):
                iou = _iou(last["box"], p["box"])
                look = _cos(last["body_embedding"], p["body_embedding"])
                face = _cos(last["face_embedding"], p["face_embedding"])
                if face is not None and face < 0.2:
                    continue
                if iou >= TRACK_IOU or (look is not None and look >= TRACK_APPEARANCE):
                    pairs.append((iou + (look or 0.0), i, j))
        used_t, used_p, nxt = set(), set(), []
        for _, i, j in sorted(pairs, reverse=True):
            if i in used_t or j in used_p:
                continue
            used_t.add(i), used_p.add(j)
            self.active[i]["detections"].append({**people[j], "t": t, "frame": self.frame_index})
            nxt.append(self.active[i])
        for j, p in enumerate(people):
            if j not in used_p:
                tr = {"id": len(self.tracks), "shot": shot,
                      "detections": [{**p, "t": t, "frame": self.frame_index}]}
                self.tracks.append(tr)
                nxt.append(tr)
        self.active = nxt


def summarize_track(track, fps=SAMPLE_FPS):
    """Average the per-frame features of one track."""
    np = _np()
    dets = track["detections"]
    faces = [(d["face_embedding"], d["face_box"][2] * d["face_box"][3])
             for d in dets if d.get("face_embedding") is not None]
    bodies = [d["body_embedding"] for d in dets if d.get("body_embedding") is not None]
    face = None
    if faces:
        v = sum(f * wgt for f, wgt in faces)
        face = v / max(1e-9, float(np.linalg.norm(v)))
    body = None
    if bodies:
        v = sum(bodies)
        body = v / max(1e-9, float(np.linalg.norm(v)))
    colours = np.mean([d["colours"] for d in dets], axis=0)
    best_face = max((d for d in dets if d.get("face_box")),
                    key=lambda d: d["face_box"][2] * d["face_box"][3], default=None)
    best_body = max(dets, key=lambda d: d["frame_share"])
    return {"id": track["id"], "shot": track["shot"],
            "start": dets[0]["t"] - 0.5 / fps, "end": dets[-1]["t"] + 0.5 / fps,
            "samples": len(dets), "frames": {d["frame"] for d in dets},
            "times": [d["t"] for d in dets],
            "face": face, "face_samples": sum(1 for d in dets if d.get("face_box")),
            "body": body, "colours": colours,
            "frame_share": float(np.mean([d["frame_share"] for d in dets])),
            "best_face": ({"t": best_face["t"], "box": best_face["face_box"]} if best_face else None),
            "best_body": {"t": best_body["t"], "box": best_body["box"]}}


# --------------------------------------------------------------- linking ----

def outfit_similarity(a, b):
    """Re-ID appearance plus named-colour overlap, 0..1-ish."""
    np = _np()
    colour = float(np.minimum(a["colours"], b["colours"]).sum())
    look = _cos(a["body"], b["body"])
    return colour if look is None else 0.8 * look + 0.2 * colour


def cluster(items, feature, similarity, threshold, cannot_link):
    """Greedy centre clustering, then merging of close groups.

    ``feature(i)`` gives a unit vector, ``cannot_link(a, b)`` says two items
    were seen at the same moment and so cannot be one person.
    """
    np = _np()
    groups = []  # {"members": [...], "sum": vector}
    for i in items:
        f = feature(i)
        best, best_sim = None, threshold
        for g in groups:
            centre = g["sum"] / max(1e-9, float(np.linalg.norm(g["sum"])))
            sim = similarity(f, centre)
            if sim >= best_sim and not any(cannot_link(i, m) for m in g["members"]):
                best, best_sim = g, sim
        if best is None:
            groups.append({"members": [i], "sum": f.copy()})
        else:
            best["members"].append(i)
            best["sum"] = best["sum"] + f
    merged = True
    while merged and len(groups) > 1:
        merged = False
        best = None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                ca = groups[a]["sum"] / max(1e-9, float(np.linalg.norm(groups[a]["sum"])))
                cb = groups[b]["sum"] / max(1e-9, float(np.linalg.norm(groups[b]["sum"])))
                sim = similarity(ca, cb)
                if sim >= threshold and (best is None or sim > best[0]):
                    if not any(cannot_link(x, y) for x in groups[a]["members"]
                               for y in groups[b]["members"]):
                        best = (sim, a, b)
        if best:
            _, a, b = best
            groups[a]["members"] += groups[b]["members"]
            groups[a]["sum"] = groups[a]["sum"] + groups[b]["sum"]
            groups.pop(b)
            merged = True
    return [g["members"] for g in groups]


def link_identities(tracks, face_threshold=FACE_THRESHOLD, outfit_threshold=OUTFIT_THRESHOLD):
    """Give each track a face group, an outfit group and a character.

    Returns (face_groups, outfit_groups, character_of) where groups are lists of
    track indexes and ``character_of[i]`` is (character key, link basis).
    """
    np = _np()
    by_frame = {}
    for i, t in enumerate(tracks):
        for f in t["frames"]:
            by_frame.setdefault(f, []).append(i)
    together = [set() for _ in tracks]
    for members in by_frame.values():
        for i in members:
            together[i].update(m for m in members if m != i)
    cannot = lambda a, b: b in together[a]

    face_items = sorted((i for i, t in enumerate(tracks) if t["face"] is not None),
                        key=lambda i: -tracks[i]["face_samples"])
    face_groups = cluster(face_items, lambda i: tracks[i]["face"], lambda a, b: float(np.dot(a, b)),
                          face_threshold, cannot)

    # Outfit vectors: re-ID embedding with the colour histogram appended.
    def outfit_vec(i):
        t = tracks[i]
        body = t["body"] if t["body"] is not None else np.zeros(1)
        colours = np.sqrt(t["colours"])
        v = np.concatenate([0.9 * body / max(1e-9, float(np.linalg.norm(body))) if body.any() else body,
                            0.45 * colours / max(1e-9, float(np.linalg.norm(colours)))])
        return v / max(1e-9, float(np.linalg.norm(v)))
    vectors = {}
    for i in range(len(tracks)):
        vectors[i] = outfit_vec(i)
    dims = {len(v) for v in vectors.values()}
    if len(dims) > 1:  # some tracks lack a body embedding: compare by colour only
        vectors = {i: np.sqrt(t["colours"]) / max(1e-9, float(np.linalg.norm(np.sqrt(t["colours"]))))
                   for i, t in enumerate(tracks)}
    outfit_items = sorted(range(len(tracks)), key=lambda i: -tracks[i]["samples"])
    outfit_groups = cluster(outfit_items, lambda i: vectors[i], lambda a, b: float(np.dot(a, b)),
                            outfit_threshold, cannot)

    character_of = {}
    for g, members in enumerate(face_groups):
        for i in members:
            character_of[i] = (("face", g), "face")
    outfit_of = {i: g for g, members in enumerate(outfit_groups) for i in members}
    for g, members in enumerate(outfit_groups):
        faceless = [i for i in members if i not in character_of]
        if not faceless:
            continue
        votes = {}
        for i in members:
            if i in character_of:
                votes[character_of[i][0]] = votes.get(character_of[i][0], 0) + tracks[i]["samples"]
        for i in faceless:
            options = [(n, key) for key, n in votes.items()
                       if not any(cannot(i, j) for j, (k, _) in character_of.items() if k == key)]
            total = sum(n for n, _ in options)
            if options and max(options)[0] >= 0.6 * total:
                character_of[i] = (max(options)[1], "clothing")
        rest = [i for i in faceless if i not in character_of]
        for i in rest:
            character_of[i] = (("outfit", g), "clothing")
    return face_groups, outfit_groups, outfit_of, character_of


# ------------------------------------------------------------ characters ----

def build_characters(tracks, face_groups, outfit_groups, outfit_of, character_of, fps=SAMPLE_FPS):
    """Turn linked tracks into numbered faces, outfits and characters."""
    np = _np()
    seconds = lambda idx: sum(tracks[i]["samples"] for i in idx) / fps
    face_id = {}
    for n, members in enumerate(sorted(face_groups, key=lambda m: -seconds(m)), 1):
        for i in members:
            face_id[i] = f"F{n}"
    outfit_id = {}
    outfit_order = sorted(range(len(outfit_groups)), key=lambda g: -seconds(outfit_groups[g]))
    outfit_name = {g: f"O{n}" for n, g in enumerate(outfit_order, 1)}
    for i, g in outfit_of.items():
        outfit_id[i] = outfit_name[g]
    members_of = {}
    for i, (key, _) in character_of.items():
        members_of.setdefault(key, []).append(i)
    order = sorted(members_of, key=lambda k: -seconds(members_of[k]))
    char_name = {key: f"C{n}" for n, key in enumerate(order, 1)}
    track_rows, characters = [], []
    for key in order:
        idx = sorted(members_of[key], key=lambda i: tracks[i]["start"])
        cid = char_name[key]
        outfits = {}
        for i in idx:
            oid = outfit_id.get(i)
            o = outfits.setdefault(oid, {"id": oid, "colours": np.zeros(len(COLOUR_NAMES)),
                                         "seconds": 0.0, "first_seconds": tracks[i]["start"],
                                         "share": -1.0, "sample": None})
            if tracks[i]["frame_share"] > o["share"]:
                o["share"], o["sample"] = tracks[i]["frame_share"], tracks[i]["best_body"]
            o["colours"] = o["colours"] + tracks[i]["colours"] * tracks[i]["samples"]
            o["seconds"] += tracks[i]["samples"] / fps
        bases = {character_of[i][1] for i in idx}
        best_face = max((tracks[i] for i in idx if tracks[i]["best_face"]),
                        key=lambda t: t["best_face"]["box"][2] * t["best_face"]["box"][3], default=None)
        best_body = max((tracks[i] for i in idx), key=lambda t: t["frame_share"])
        characters.append({
            "id": cid,
            "identified_by": "face" if key[0] == "face" else "clothing",
            "face_ids": sorted({face_id[i] for i in idx if i in face_id}, key=lambda s: int(s[1:])),
            "outfit_ids": [o["id"] for o in sorted(outfits.values(), key=lambda o: o["first_seconds"])],
            "outfits": [{"id": o["id"], "colours": colour_words(o["colours"] / max(1e-9, o["colours"].sum())),
                         "seconds": _r(o["seconds"], 1), "first_seconds": _r(o["first_seconds"], 1),
                         "sample": o["sample"]}
                        for o in sorted(outfits.values(), key=lambda o: o["first_seconds"])],
            "links": sorted(bases),
            "screen_seconds": _r(seconds(idx), 1),
            "face_visible_seconds": _r(sum(tracks[i]["face_samples"] for i in idx) / fps, 1),
            "appearances": len(idx),
            "shots": sorted({tracks[i]["shot"] for i in idx}),
            "first_seconds": _r(tracks[idx[0]]["start"], 1),
            "last_seconds": _r(max(tracks[i]["end"] for i in idx), 1),
            "mean_frame_share": _r(float(np.mean([tracks[i]["frame_share"] for i in idx])), 3),
            "best_face": best_face["best_face"] if best_face else None,
            "best_body": best_body["best_body"]})
        for i in idx:
            track_rows.append({"track": f"T{tracks[i]['id'] + 1}", "character": cid,
                               "face_id": face_id.get(i), "outfit_id": outfit_id.get(i),
                               "link": character_of[i][1],
                               "start_seconds": _r(tracks[i]["start"], 1),
                               "end_seconds": _r(tracks[i]["end"], 1),
                               "shot": tracks[i]["shot"], "times": tracks[i]["times"],
                               "face_visible": tracks[i]["face_samples"] > 0})
    track_rows.sort(key=lambda r: (r["start_seconds"], r["character"]))
    return characters, track_rows


def attach_to_timeline(timeline, track_rows, fps=SAMPLE_FPS):
    """Each interval lists who is on screen and for how long."""
    for window in timeline:
        a, b = window["start_seconds"], window["end_seconds"]
        present = {}
        for row in track_rows:
            inside = [t for t in row["times"] if a <= t < b]
            if inside:
                p = present.setdefault(row["character"], {"id": row["character"], "seconds": 0.0,
                                                          "face_visible": False})
                p["seconds"] += len(inside) / fps
                p["face_visible"] = p["face_visible"] or row["face_visible"]
        window["characters_present"] = sorted(
            ({**p, "seconds": _r(p["seconds"], 1)} for p in present.values()),
            key=lambda p: -p["seconds"])


def link_speech(speech, track_rows, characters):
    """Tie voice groups and lines to characters seen on screen while they are spoken."""
    votes = {}
    by_id = {c["id"]: c for c in characters}
    for seg in speech:
        a, b = seg["start_seconds"], seg["end_seconds"]
        visible = {}
        for row in track_rows:
            n = sum(1 for t in row["times"] if a <= t < b)
            if n and row["face_visible"]:
                visible[row["character"]] = visible.get(row["character"], 0) + n
        seg["_visible"] = visible
        if len(visible) == 1 and seg.get("speaker"):
            (cid,) = visible
            votes.setdefault(seg["speaker"], {}).setdefault(cid, 0)
            votes[seg["speaker"]][cid] += 1
    voice_of = {}
    for speaker, counts in votes.items():
        cid, n = max(counts.items(), key=lambda kv: kv[1])
        if n >= 2 and n >= 0.5 * sum(counts.values()):
            voice_of[speaker] = cid
            by_id[cid].setdefault("voice_groups", []).append(speaker)
    for seg in speech:
        visible = seg.pop("_visible", {})
        cid, source = None, None
        if seg.get("speaker") in voice_of:
            cid, source = voice_of[seg["speaker"]], "voice_group"
        elif len(visible) == 1:
            cid, source = next(iter(visible)), "only_face_on_screen"
        if cid:
            seg["character"], seg["character_source"] = cid, source
            c = by_id[cid]
            c["dialogue_seconds"] = _r((c.get("dialogue_seconds") or 0.0)
                                       + seg["end_seconds"] - seg["start_seconds"], 1)
    for c in characters:
        c.setdefault("voice_groups", [])
        c.setdefault("dialogue_seconds", 0.0)


def rank_prominence(characters):
    """Screen time, dialogue and size relative to the most prominent character."""
    if not characters:
        return
    top = lambda key: max((c[key] or 0.0) for c in characters) or 1.0
    s, d, a = top("screen_seconds"), top("dialogue_seconds"), top("mean_frame_share")
    for c in characters:
        c["prominence"] = _r(0.5 * c["screen_seconds"] / s + 0.3 * (c["dialogue_seconds"] or 0) / d
                             + 0.2 * c["mean_frame_share"] / a)
    peak = max(c["prominence"] for c in characters) or 1.0
    for c in characters:
        c["prominence"] = _r(c["prominence"] / peak)
    ranked = sorted(characters, key=lambda c: -c["prominence"])
    for n, c in enumerate(ranked, 1):
        c["prominence_rank"] = n
        c["likely_main_character"] = n == 1 or c["prominence"] >= 0.8


def analyze_characters(video, info, shots, timeline, speech, *, fps=SAMPLE_FPS,
                       backend=None, frame_loader=None, min_screen_seconds=None):
    """Find, follow and link people. Adds to timeline and speech in place."""
    detect = backend or OpenCVPeople()
    frames = frame_loader() if frame_loader else sample_frames(video, info, fps)
    tracker = Tracker(fps)
    count = 0
    try:
        for t, frame in frames:
            tracker.update(t, shot_index(shots, t), detect(frame))
            count += 1
    finally:
        close = getattr(detect, "close", None)
        if close:
            close()
    min_seconds = (2.0 / fps) if min_screen_seconds is None else min_screen_seconds
    tracks = [summarize_track(tr, fps) for tr in tracker.tracks]
    # Single glimpses without a face are too weak to link.
    tracks = [t for t in tracks if t["samples"] / fps >= min_seconds or t["face"] is not None]
    for n, t in enumerate(tracks):
        t["id"] = n
    face_groups, outfit_groups, outfit_of, character_of = link_identities(tracks)
    characters, rows = build_characters(tracks, face_groups, outfit_groups, outfit_of, character_of, fps)
    attach_to_timeline(timeline, rows, fps)
    link_speech(speech, rows, characters)
    rank_prominence(characters)
    for r in rows:
        r.pop("times")
    print(f"Characters: {count} frames, {len(tracks)} appearances, {len(characters)} characters "
          f"({len(face_groups)} faces, {len(outfit_groups)} outfits)", file=sys.stderr)
    return {"characters": characters, "character_tracks": rows,
            "settings": {"sample_fps": fps, "frames": count, "appearances": len(tracks),
                         "face_groups": len(face_groups), "outfit_groups": len(outfit_groups),
                         "face_threshold": FACE_THRESHOLD, "outfit_threshold": OUTFIT_THRESHOLD,
                         "models": {k: Path(v[0]).name for k, v in MODELS.items()},
                         "method": "yolox_yunet_sface_youtureid_tracks_v1"}}
