#!/usr/bin/env python3


from ultralytics import YOLO
import os
import yaml

def create_dataset_yaml():
    """
    Resolve data path relative to this script's location.
    Works on any machine without hardcoding username or workspace path.
    """
    # This script is at: src/robot_perception/panda_perception/train_custom_model.py
    # Data is at:        src/robot_perception/data/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path  = os.path.abspath(os.path.join(script_dir, '..', 'data'))
    yaml_path  = os.path.join(data_path, 'dataset.yaml')

    config = {
        'path':  data_path,
        'train': 'images/train',
        'val':   'images/val',
        'nc':    4,
        'names': {
            0: 'coke_can',
            1: 'cricket_ball',
            2: 'small_box',
            3: 'wood_cube',
           
        }
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f'dataset.yaml → {yaml_path}')
    return yaml_path


def train():
    dataset_yaml = create_dataset_yaml()

    # Load pretrained YOLOv8n as starting point (transfer learning)
    # Transfer learning means we start with weights already trained on 80 COCO classes.
    # This means we need FAR fewer images than training from scratch.
    # The network already knows how to detect edges, shapes, textures.
    # We only teach it the NEW classes (cube, small_box) on top.
    model = YOLO('yolov8n.pt')

    print("Starting training...")
    print("This will take 2-3 hours on CPU. You can stop and resume anytime.")
    print()

    results = model.train(
        data=dataset_yaml,

        # Number of complete passes through the training data.
        # 50 is enough for transfer learning with ~200 images.
        epochs=50,

        # Input image size. 640 is standard for YOLOv8.
        imgsz=640,

        # Images processed together per training step.
        # 8 is safe for 8GB RAM. Lower if you get memory errors.
        batch=8,

        # Use CPU — no GPU available on your laptop.
        device='cpu',

        # Stop training early if no improvement for 10 epochs.
        # Saves time and prevents overfitting.
        patience=10,

        # Where to save results
        project='panda_yolo_training',
        name='custom_v1',

        # Save the best model checkpoint (not just the last one)
        save=True,

        # Print training progress every epoch
        verbose=True,
    )

    # The best model is saved at:
    best_model_path = 'panda_yolo_training/custom_v1/weights/best.pt'
    print()
    print(f"Training complete!")
    print(f"Best model saved to: {best_model_path}")
    print()
    print("To use your custom model, update perception.launch.py:")
    print(f"  'model_path': '{os.path.abspath(best_model_path)}',")

    return results


def validate_model():
    """Quick validation to check model accuracy after training."""
    best_path = 'panda_yolo_training/custom_v1/weights/best.pt'

    if not os.path.exists(best_path):
        print("No trained model found. Run train() first.")
        return

    model = YOLO(best_path)

    # Run validation on the val set
    # mAP50 > 0.8 means excellent detection accuracy
    # mAP50 > 0.6 means acceptable
    metrics = model.val()

    print(f"\nValidation Results:")
    print(f"  mAP50      : {metrics.box.map50:.3f}  (target: > 0.80)")
    print(f"  mAP50-95   : {metrics.box.map:.3f}")
    print(f"  Precision  : {metrics.box.mp:.3f}")
    print(f"  Recall     : {metrics.box.mr:.3f}")


if __name__ == '__main__':
    print("=" * 60)
    print("YOLOv8 Custom Training — Franka Panda Sorting")
    print("=" * 60)
    print()
    print("Make sure you have labelled images in ~/training_data/")
    print("before starting. See docstring at top of this file.")
    print()

    import sys
    if '--validate' in sys.argv:
        validate_model()
    else:
        train()
