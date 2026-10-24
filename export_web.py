"""Export the pretrained model's weights for the browser version (web/)."""
import json
import os
import shutil

import numpy as np

from model import HERE, TARGETS, load_model

WEB = os.path.join(HERE, "web")

model, labels = load_model()
manifest, chunks, offset = {}, [], 0
for name, t in model.state_dict().items():
    arr = t.detach().numpy().astype("<f4").ravel()
    manifest[name] = {"shape": list(t.shape), "offset": offset, "length": arr.size}
    chunks.append(arr)
    offset += arr.size

np.concatenate(chunks).tofile(os.path.join(WEB, "weights.bin"))
with open(os.path.join(WEB, "model.json"), "w") as f:
    json.dump({"layers": manifest, "labels": labels, "targets": TARGETS}, f)
shutil.copy(os.path.join(HERE, "hand_landmarker.task"), WEB)
print(f"exported {offset} weights, {len(labels)} labels")
