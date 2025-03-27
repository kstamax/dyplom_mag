import argparse
import sys
from pathlib import Path
from dyplom_mag.mean_teacher_2.sf_yolo_ultralytics import SFYOLOTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Source-Free Domain Adaptation for YOLO Object Detection")

    # Main parameters
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory containing data.yaml')
    parser.add_argument('--weights', type=str, required=True, help='Path to initial teacher model weights')
    parser.add_argument('--img-size', type=int, default=640, help='Image size for training')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    
    # SF-YOLO specific parameters
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold for pseudo-labels')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IoU threshold for NMS')
    parser.add_argument('--teacher-alpha', type=float, default=0.999, help='Teacher EMA decay rate')
    parser.add_argument('--ssm-alpha', type=float, default=0.5, help='SSM weight transfer rate')
    parser.add_argument('--style-alpha', type=float, default=0.2, help='Style transfer intensity')
    parser.add_argument('--max-gt-boxes', type=int, default=20, help='Maximum number of ground truth boxes')
    
    # Style transfer parameters
    parser.add_argument('--style-path', type=str, default="", help='Path to style images directory')
    
    # Output parameters
    parser.add_argument('--project', type=str, default='runs/train', help='Project directory')
    parser.add_argument('--name', type=str, default='sf-yolo', help='Experiment name')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create save directory
    save_dir = Path(args.project) / args.name
    
    # Initialize and run the trainer
    trainer = SFYOLOTrainer(
        data_dir=args.data_dir,
        teacher_model_path=args.weights,
        style_path=args.style_path,
        device=args.device,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        teacher_alpha=args.teacher_alpha,
        ssm_alpha=args.ssm_alpha,
        style_alpha=args.style_alpha,
        max_gt_boxes=args.max_gt_boxes,
        save_dir=save_dir,
    )
    
    # Run training
    teacher_model, student_model, metrics_history = trainer.train()
    
    # Print final results
    print(f"Training completed.")
    print(f"Best model saved to {save_dir}/weights/best_teacher.pt")
    print(f"Final model saved to {save_dir}/weights/last_teacher.pt")
    
    # Return success
    return 0

if __name__ == "__main__":
    sys.exit(main())