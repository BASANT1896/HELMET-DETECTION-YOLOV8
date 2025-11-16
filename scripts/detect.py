# detect.py
from ultralytics import YOLO


def run_inference(
    weights="best.pt",
    source="test/images",
    project="runs",
    name="spd_ema_predictions",
    conf=0.25
):
    """
    Generate predictions using trained SPD + EMA YOLOv8 model.
    """
    print(" Loading model:", weights)
    model = YOLO(weights)

    print(" Running inference...")
    model.predict(
        source=source,
        save=True,
        conf=conf,
        project=project,
        name=name,
        exist_ok=True
    )

    print(" Predictions saved!")


if __name__ == "__main__":
    run_inference()
