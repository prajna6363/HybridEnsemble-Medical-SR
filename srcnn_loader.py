# srcnn_loader.py
import os
import torch
import importlib.util

# === Edit these if your files are elsewhere ===
SRCNN_PY = "models/srcnn.py"
SRCNN_PTH = "weights/srcnn.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _import_module_from_path(path, module_name="user_srcnn"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"SRCNN python file not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def detect_scale_from_ckpt(ckpt):
    # Try common keys
    if isinstance(ckpt, dict):
        if "scale" in ckpt:
            return int(ckpt["scale"])
        if "args" in ckpt and isinstance(ckpt["args"], dict) and "scale" in ckpt["args"]:
            return int(ckpt["args"]["scale"])
        if "model_state" in ckpt:
            # fall through
            pass
    return None

def load_srcnn(srcnn_py=SRCNN_PY, weights_pth=SRCNN_PTH, device=DEVICE):
    mod = _import_module_from_path(srcnn_py)
    # find candidate class names
    cls = None
    for name in ("MedicalSRCNN_Plus", "SRCNNPlus", "SRCNN", "Srcnn", "Net", "Model"):
        if hasattr(mod, name):
            cls = getattr(mod, name)
            break
    # also accept a factory function
    factory = None
    for name in ("build_model","create_model","make_model"):
        if hasattr(mod, name):
            factory = getattr(mod, name)
            break

    if cls is None and factory is None:
        # fallback: if module defines a 'model' variable
        if hasattr(mod, "model"):
            model = getattr(mod, "model")
        else:
            raise RuntimeError("Could not find SRCNN class or factory in the SRCNN python file.")
    else:
        if factory is not None:
            model = factory()
        else:
            model = cls()

    model = model.to(device)
    scale = None
    # load weights if present
    if weights_pth and os.path.exists(weights_pth):
        ckpt = torch.load(weights_pth, map_location="cpu")
        # try to detect scale
        detected = detect_scale_from_ckpt(ckpt)
        if detected is not None:
            scale = detected
        # state dict handling
        sd = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
        if isinstance(sd, dict):
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
        try:
            model.load_state_dict(sd, strict=False)
            print("✔ SRCNN weights loaded from", weights_pth)
        except Exception as e:
            print("! Warning: SRCNN weights load failed (attempting partial load):", e)
            try:
                model.load_state_dict(sd, strict=False)
            except Exception:
                pass
    else:
        print("! SRCNN weights not found at", weights_pth)

    # attempt to fetch attribute scale from model
    try:
        if hasattr(model, "scale_factor"):
            scale = int(getattr(model, "scale_factor"))
        elif hasattr(model, "scale"):
            scale = int(getattr(model, "scale"))
    except Exception:
        pass

    if scale is None:
        # default to 2 (as you suspected). If wrong, ensemble script will upsample appropriately.
        scale = 2
        print("⚠ SRCNN scale not detected — defaulting to x2. If this is wrong, update the loader.")

    model.eval()
    return model, scale

if __name__ == "__main__":
    m, s = load_srcnn()
    print("SRCNN model loaded. Detected scale:", s)
