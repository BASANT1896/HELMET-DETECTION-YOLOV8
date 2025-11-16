# train.py
import os
from ultralytics import YOLO
import torch.nn as nn

from spd import replace_conv_with_spd
from ema import EMAAttention


def train(
    model_path="yolov8n.pt",
    data_yaml="data.yaml",
    epochs=20,
    batch=16,
    imgsz=640,
    project="spd_ema_runs",
    name="train"
):
    """
    Train YOLOv8 with SPDConv + EMA Attention integration.
    """

    # Disable WandB
    os.environ["WANDB_MODE"] = "disabled"
    import ultralytics
    ultralytics.utils.callbacks.wb = {}

    print(" Loading model...")
    model = YOLO(model_path)

    print(" Replacing Conv layers with SPDConv...")
    replace_conv_with_spd(model.model, use_modulation=True)

    print(" Adding EMA Attention block...")
    model.model.model[2] = nn.Sequential(
        model.model.model[2],
        EMAAttention(64)
    )

    print(" Starting training...\n")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        exist_ok=True
    )

    print(" Training completed.")
    return results


if __name__ == "__main__":
    train()
