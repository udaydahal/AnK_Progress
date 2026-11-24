"""Give every face of every MFCAD++ part a visual feature vector from a frozen SigLIP2 image encoder.

For each part:
  1. render 14 views (render_views.py): a grey picture + a face-id buffer per view
  2. the encoder turns each picture into a 24 x 24 grid of patches, 768 numbers per patch
  3. that grid is smoothly enlarged to 96 x 96 cells (4 x 4 pixels each), so two small faces that sit
     in the same 16 x 16 patch still get different vectors
  4. for every face, average the cell vectors lying under its pixels (a cell counts as much as the face
     covers of it), over all views that see the face
Result per part: (n_faces, 769) float16 = 768 visual numbers + 1 flag "some view sees this face"
(a face hidden in every view gets all zeros). Row i belongs to node i of the part's graph.

Run:  python vlm_features.py [train_n] [val_n] [test_n]     (defaults 6000 1000 1500, as train_mfcad.py)
Out:  data/feature_cache/{split}_{n}_{views}v_pool{POOL}.pt
"""
import os
import sys
import time
from multiprocessing import Pool

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                        # shared code: brep_gnn/
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "mfcad"))  # split_files

from render_views import IMG_SIZE, VIEWS, render_for_features

MODEL = "google/siglip2-base-patch16-384"
PATCH = 16
GRID = IMG_SIZE // PATCH            # 24 x 24 patches from the encoder
POOL = 4                            # pixels per side of the finer cells the faces are averaged over
FINE = IMG_SIZE // POOL             # 96 x 96 cells
CACHE = os.path.join(HERE, "data", "feature_cache")
CHUNK = 64                          # parts rendered at a time before the GPU encodes them


def load_encoder(device):
    from transformers import AutoModel
    return AutoModel.from_pretrained(MODEL).vision_model.eval().half().to(device)


@torch.no_grad()
def encode(encoder, shaded, device):
    """(views, H, W, 3) uint8 pictures -> (views, GRID*GRID, 768) patch vectors."""
    x = torch.from_numpy(shaded).to(device).permute(0, 3, 1, 2).half() / 255
    return encoder(pixel_values=(x - 0.5) / 0.5).last_hidden_state.float()


def face_features(tokens, face_id, n_faces):
    """Average the (smoothly enlarged) patch vectors under each face. -> (n_faces, 769)"""
    views = len(face_id)
    grid = tokens.transpose(1, 2).reshape(views, -1, GRID, GRID)                  # (views, 768, 24, 24)
    cells = F.interpolate(grid, size=(FINE, FINE), mode="bilinear", align_corners=False)
    cells = cells.flatten(2).transpose(1, 2)                                     # (views, 96*96, 768)
    ids = torch.from_numpy(face_id.astype("int64")).to(tokens.device)
    ids = ids.view(views, FINE, POOL, FINE, POOL).permute(0, 1, 3, 2, 4).reshape(views, FINE * FINE, POOL * POOL)
    counts = torch.zeros(views, FINE * FINE, n_faces + 1, device=tokens.device)
    counts.scatter_add_(2, ids, torch.ones_like(ids, dtype=torch.float))   # pixels of each face in each cell
    counts = counts[:, :, 1:]                                              # drop the background
    seen = counts.sum((0, 1))                                              # pixels of each face, all views
    mean = torch.einsum("vcf,vcd->fd", counts, cells) / seen.clamp(min=1)[:, None]
    return torch.cat([mean, (seen > 0).float()[:, None]], dim=1)


def cache_path(split, n):
    return os.path.join(CACHE, f"{split}_{n}_{len(VIEWS)}v_pool{POOL}.pt")


def build_split(split, count):
    from train_mfcad import split_files
    folder, files = split_files(split, count)
    cache_file = cache_path(split, len(files))
    if os.path.exists(cache_file):
        return torch.load(cache_file, weights_only=False)

    os.makedirs(CACHE, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = load_encoder(device)
    paths = [os.path.join(folder, f) for f in files]
    print(f"encoding {len(paths)} {split} parts, {len(VIEWS)} views each ...")
    start, feats, failed = time.time(), [], 0
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        for i in range(0, len(paths), CHUNK):
            for r in pool.map(render_for_features, paths[i:i + CHUNK]):
                if r is None:
                    feats.append(None)
                    failed += 1
                    continue
                tokens = encode(encoder, r["shaded"], device)
                feats.append(face_features(tokens, r["face_id"], r["n_faces"]).half().cpu())
            print(f"  {len(feats)}/{len(paths)} ({time.time() - start:.0f}s)")
    print(f"  done, {failed} parts could not be rendered ({time.time() - start:.0f}s)")
    torch.save(feats, cache_file)
    return feats


if __name__ == "__main__":
    n_train, n_val, n_test = ([int(a) for a in sys.argv[1:]] + [6000, 1000, 1500][len(sys.argv) - 1:])[:3]
    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        build_split(split, n)
