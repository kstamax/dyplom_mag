import os
import copy
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.torch_utils import ModelEMA, de_parallel
from ultralytics.utils.ops import non_max_suppression, xyxy2xywhn
from dyplom_mag.target_augment.enhance_style import get_style_images
from dyplom_mag.target_augment.enhance_vgg16 import enhance_vgg16
from dyplom_mag.mean_teacher_2.sf_yolo_loss import SFYOLOv8Loss

BASE_DIR = os.path.dirname(__file__)


class WeightEMA(torch.optim.Optimizer):
    """
    Exponential moving average weight optimizer for Mean Teacher model.
    
    This optimizer updates teacher model parameters using EMA of student parameters.
    """
    def __init__(self, teacher_params, student_params, alpha=0.999):
        """
        Initialize WeightEMA optimizer.
        
        Args:
            teacher_params: parameters of the teacher model
            student_params: parameters of the student model
            alpha: EMA decay rate
        """
        self.teacher_params = list(teacher_params)
        self.student_params = list(student_params)
        self.alpha = alpha
        
        if len(self.teacher_params) != len(self.student_params):
            raise ValueError("Teacher and student parameter lengths don't match")
        
        defaults = dict()
        super(WeightEMA, self).__init__(self.teacher_params, defaults)

    def step(self):
        """
        Update teacher model parameters with EMA of student parameters.
        """
        for teacher_param, student_param in zip(self.teacher_params, self.student_params):
            teacher_param.data.mul_(self.alpha).add_(student_param.data, alpha=1 - self.alpha)


class SFYOLOTrainer:
    """
    Source-Free YOLO Trainer implementing the paper's approach with Ultralytics YOLO.
    """
    def __init__(
        self,
        data_dir,
        teacher_model_path,
        style_path="",
        device="cuda",
        img_size=640,
        batch_size=16,
        epochs=30,
        conf_thres=0.25,
        iou_thres=0.45,
        teacher_alpha=0.999,
        ssm_alpha=0.5,
        style_alpha=0.2,
        max_gt_boxes=20,
        save_dir="runs/train/sf-yolo",
        encoder_path="",
        decoder_path="",
        fc1_path="",
        fc2_path="",
        save_style_samples=False,
    ):
        """
        Initialize the Source-Free YOLO Trainer.
        
        Args:
            data_dir: Directory containing the dataset
            teacher_model_path: Path to the teacher model weights
            style_path: Directory containing style images
            device: Device to use (cuda or cpu)
            img_size: Image size for training
            batch_size: Batch size for training
            epochs: Number of epochs to train
            conf_thres: Confidence threshold for pseudo-labels
            iou_thres: IoU threshold for NMS
            teacher_alpha: Alpha value for teacher EMA update
            ssm_alpha: Alpha value for SSM weight transfer
            style_alpha: Alpha value for style transfer
            max_gt_boxes: Maximum number of ground truth boxes
            save_dir: Directory to save results
            encoder_path: Path to the encoder weights for style transfer
            decoder_path: Path to the decoder weights for style transfer
            fc1_path: Path to the fc1 weights for style transfer
            fc2_path: Path to the fc2 weights for style transfer
            save_style_samples: Whether to save style samples for debugging
        """
        self.data_dir = data_dir
        self.device = device
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.teacher_alpha = teacher_alpha
        self.ssm_alpha = ssm_alpha
        self.style_alpha = style_alpha
        self.max_gt_boxes = max_gt_boxes
        self.save_dir = Path(save_dir)
        self.style_path = style_path
        if encoder_path:
            self.encoder_path = encoder_path
        else:
            self.encoder_path = os.path.join(BASE_DIR, "..", "target_augment", "pre_trained", "vgg16_ori.pth")
        if decoder_path:
            self.decoder_path = decoder_path
        else:
            self.decoder_path = os.path.join(BASE_DIR, "..", "target_augment", "models", "decoder.pth")
        if fc1_path:
            self.fc1_path = fc1_path
        else:
            os.path.join(BASE_DIR, "..", "target_augment", "models", "fc1.pth")
        if fc2_path:
            self.fc2_path = fc2_path
        else:
            self.fc2_path = os.path.join(BASE_DIR, "..", "target_augment", "models", "fc2.pth")

        self.save_style_samples = save_style_samples

        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create weights directory
        self.weights_dir = self.save_dir / "weights"
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        
        # Load teacher model
        print(f"Loading teacher model from {teacher_model_path}")
        self.teacher_model = YOLO(teacher_model_path)
        
        # Create student model as a copy of teacher
        print("Creating student model")
        self.student_model = copy.deepcopy(self.teacher_model)
        
        # Find data.yaml file
        self.data_yaml = os.path.join(data_dir, "data.yaml")
        if not os.path.exists(self.data_yaml):
            raise FileNotFoundError(f"Could not find data.yaml in {data_dir}")
        
        # Initialize style transfer model
        self.style_transfer = None
        
        # Variables for tracking training progress
        self.best_map = 0
        self.best_epoch = 0
        self.best_teacher_weights = None
        self.best_student_weights = None
        self.metrics_history = []
        
    def init_style_transfer(self):
        """Initialize the style transfer model"""
        # Create a class with attributes needed by the style transfer function
        class StyleOpt:
            def __init__(self, style_path, img_size, style_alpha, encoder_path, decoder_path, fc1_path, fc2_path, save_style_samples):
                self.style_path = style_path
                self.imgsz = img_size
                self.style_add_alpha = style_alpha
                self.random_style = style_path == ""
                self.cuda = torch.cuda.is_available()
                self.log_dir = "./enhance_style_samples"
                self.encoder_path = encoder_path
                self.decoder_path = decoder_path
                self.fc1 = fc1_path
                self.fc2 = fc2_path
                self.save_style_samples = save_style_samples
                
        opt = StyleOpt(
            self.style_path, 
            self.img_size, 
            self.style_alpha,
            self.encoder_path,
            self.decoder_path,
            self.fc1_path,
            self.fc2_path,
            self.save_style_samples
        )
        self.style_transfer = enhance_vgg16(opt)
        
    def train(self):
        """Main training loop following the paper's approach"""
        # Initialize style transfer
        self.init_style_transfer()
        
        # Create a custom data loader from the training data
        train_loader = self._create_dataloader()
        
        # Initialize EMA for the student model (for potential use)
        ema = ModelEMA(self.student_model.model)
        
        # Get model parameters for EMA optimizer
        teacher_params = [p for p in self.teacher_model.model.parameters() if p.requires_grad]
        student_params = [p for p in self.student_model.model.parameters() if p.requires_grad]
        
        # Initialize optimizer for teacher (EMA)
        teacher_optimizer = WeightEMA(teacher_params, student_params, alpha=self.teacher_alpha)
        
        # Initialize optimizer for student (SGD or Adam)
        student_optimizer = torch.optim.Adam(student_params, lr=0.001)
        
        # Learning rate scheduler
        def cosine_lr_scheduler(epoch):
            return 0.5 * (1 + np.cos(np.pi * epoch / self.epochs))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(student_optimizer, lr_lambda=cosine_lr_scheduler)
        
        # Initialize the compute loss function
        compute_loss = self._get_compute_loss()
        
        print(f"Starting Source-Free YOLO training for {self.epochs} epochs")
        t0 = time.time()
        
        # Main epoch loop
        for epoch in range(self.epochs):
            self.teacher_model.model.train()
            self.student_model.model.train()
            
            # Update student with teacher weights using SSM (after first epoch)
            if epoch > 0 and self.ssm_alpha > 0:
                student_state_dict = self.student_model.model.state_dict()
                teacher_state_dict = self.teacher_model.model.state_dict()
                
                for name, param in student_state_dict.items():
                    if name in teacher_state_dict:
                        param.data.copy_((1.0 - self.ssm_alpha) * param.data + 
                                        self.ssm_alpha * teacher_state_dict[name].data)
                print(f"Updated student model with teacher weights (alpha={self.ssm_alpha})")
            
            # Track losses
            mloss = torch.zeros(3, device=self.device)  # box, obj, cls
            
            # Set up progress bar
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                       desc=f"Epoch {epoch+1}/{self.epochs}")
            
            # Batch loop
            for i, (imgs, targets, paths, _) in pbar:
                # Convert images to the format expected by the models
                imgs_255 = imgs.clone().to(torch.float32).to(self.device)
                imgs = imgs.to(self.device).float() / 255.0
                
                # Generate styled images using AdaIN
                imgs_style = get_style_images(imgs_255, opt=None, adain=self.style_transfer) / 255.0
                
                # Teacher forward pass to generate pseudo-labels
                with torch.no_grad():
                    self.teacher_model.model.eval()  # Set to eval mode for inference
                    teacher_output = self.teacher_model.model(imgs)
                    
                    # Apply NMS to teacher predictions for pseudo-labeling
                    teacher_predictions = non_max_suppression(
                        teacher_output[0] if isinstance(teacher_output, tuple) else teacher_output,
                        conf_thres=self.conf_thres,
                        iou_thres=self.iou_thres,
                        max_det=self.max_gt_boxes
                    )
                
                # Create pseudo-labels from teacher predictions
                batch_size, ch, height, width = imgs.shape
                pseudo_labels = []
                
                for img_idx, pred in enumerate(teacher_predictions):
                    if len(pred):
                        # Format: [cls, x, y, w, h]
                        # First extract class and bbox coordinates
                        cls_data = pred[:, 5:6]  # class index
                        bbox_data = pred[:, :4]  # xyxy format
                        
                        # Convert from xyxy to normalized xywh format
                        bbox_data_norm = xyxy2xywhn(bbox_data, w=width, h=height)
                        
                        # Combine class and normalized bbox
                        combined = torch.cat((cls_data, bbox_data_norm), dim=1)
                        
                        # Add image index as first column
                        img_indices = torch.full((combined.shape[0], 1), img_idx, device=self.device)
                        pseudo_label = torch.cat((img_indices, combined), dim=1)
                        
                        pseudo_labels.append(pseudo_label)
                
                # If no valid pseudo-labels were created, create a dummy one
                if not pseudo_labels:
                    # Create a dummy pseudo-label to avoid errors
                    pseudo_labels = [torch.tensor([[0, 0, 0.5, 0.5, 0.1, 0.1]], device=self.device)]
                
                # Concatenate all pseudo-labels
                pseudo_labels = torch.cat(pseudo_labels, dim=0)
                
                # Student forward pass on styled images
                student_optimizer.zero_grad()
                student_output = self.student_model.model(imgs_style)
                
                # Compute loss using pseudo-labels
                loss, loss_items = compute_loss(student_output, pseudo_labels, self.teacher_model, self.student_model)
                
                # Backward pass and optimizer step for student
                loss.backward()
                student_optimizer.step()
                
                # Update teacher with EMA of student
                teacher_optimizer.step()
                
                # Update progress bar
                mloss = (mloss * i + loss_items) / (i + 1)
                mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"
                pbar.set_description(
                    f"Epoch {epoch+1}/{self.epochs} | {mem} | box: {mloss[0]:.4g}, obj: {mloss[1]:.4g}, cls: {mloss[2]:.4g}"
                )
            
            # End of epoch
            scheduler.step()
            
            # Validate and save models
            metrics = self._validate()
            self.metrics_history.append(metrics)
            
            current_map = metrics.get('metrics/mAP50-95', 0)
            print(f"Epoch {epoch+1} results - mAP50-95: {current_map:.4f}, mAP50: {metrics.get('metrics/mAP50', 0):.4f}")
            
            # Save last models
            self.teacher_model.save(self.weights_dir / f"last_teacher.pt")
            self.student_model.save(self.weights_dir / f"last_student.pt")
            
            # Check if this is the best model so far
            if current_map > self.best_map:
                self.best_map = current_map
                self.best_epoch = epoch
                
                # Save best models
                self.teacher_model.save(self.weights_dir / f"best_teacher.pt")
                self.student_model.save(self.weights_dir / f"best_student.pt")
                
                # Store best weights
                self.best_teacher_weights = copy.deepcopy(self.teacher_model.model.state_dict())
                self.best_student_weights = copy.deepcopy(self.student_model.model.state_dict())
                
                print(f"New best model saved with mAP50-95: {current_map:.4f}")
        
        # Training complete
        total_time = time.time() - t0
        print(f"Training completed in {total_time/3600:.3f} hours")
        print(f"Best model was at epoch {self.best_epoch+1} with mAP50-95: {self.best_map:.4f}")
        
        # Restore best models if available
        if self.best_teacher_weights is not None:
            self.teacher_model.model.load_state_dict(self.best_teacher_weights)
            self.student_model.model.load_state_dict(self.best_student_weights)
            print("Restored best model weights")
        
        # Final validation
        print("Running final validation...")
        final_metrics = self._validate()
        print("Final metrics:")
        for k, v in final_metrics.items():
            print(f"  {k}: {v:.4f}")
        
        # Save final results summary
        self._save_results()
        
        return self.teacher_model, self.student_model, self.metrics_history
    
    def _create_dataloader(self):
        """Create a dataloader from the Ultralytics dataset"""
        from ultralytics.data.build import build_dataloader
        
        # Load the data.yaml configuration
        with open(self.data_yaml, errors='ignore') as f:
            data_dict = yaml.safe_load(f)
        
        # Get the training data path
        train_path = data_dict['train']
        
        # Build the dataloader
        loader = build_dataloader(
            path=train_path,
            imgsz=self.img_size,
            batch_size=self.batch_size,
            stride=32,  # Assuming YOLO with stride 32
            hyp=None,  # No hyperparameters for now
            augment=True,
            cache=False,
            pad=0.0,
            rect=False,
            rank=-1,
            workers=8,
            seed=0,
            close_mosaic=0
        )[0]
        
        return loader
    
    def _get_compute_loss(self):
        """Get the loss computation function for YOLO"""
        try:
            # Use our custom SFYOLOv8Loss implementation
            loss_fn = SFYOLOv8Loss(self.student_model.model)
            
            def compute_loss(outputs, targets, teacher_model=None, student_model=None):
                return loss_fn(outputs, targets, teacher_model, student_model)
                
            return compute_loss
            
        except Exception as e:
            print(f"Warning: Could not initialize SFYOLOv8Loss: {e}")
            print("Falling back to built-in loss function")
            
            # Try to access the built-in loss function
            try:
                from ultralytics.models.yolo.detect import DetectionLoss
                loss_fn = DetectionLoss(self.student_model.model)
                
                # Create a wrapper to match the expected interface
                def compute_loss(outputs, targets, teacher_model=None, student_model=None):
                    # Format targets to match what DetectionLoss expects
                    batch = self._format_pseudo_labels_for_detection_loss(targets)
                    return loss_fn(outputs, batch)
                    
                return compute_loss
                
            except ImportError:
                # Fallback to a very simple loss function
                print("Warning: Could not import DetectionLoss, using simplified loss function")
                
                def compute_loss(outputs, targets, teacher_model=None, student_model=None):
                    # This is a placeholder - just using a simple MSE loss
                    if isinstance(outputs, tuple):
                        pred = outputs[0]
                    else:
                        pred = outputs
                    
                    # Initialize losses
                    box_loss = torch.tensor(0.0, device=self.device)
                    obj_loss = torch.tensor(0.0, device=self.device)
                    cls_loss = torch.tensor(0.0, device=self.device)
                    
                    # Very simplified loss - not recommended for actual training
                    total_loss = torch.mean(pred**2) * 0.01  # Just to have something
                    
                    return total_loss, torch.tensor([box_loss, obj_loss, cls_loss], device=self.device)
                
                return compute_loss
    
    def _format_pseudo_labels_for_detection_loss(self, pseudo_labels):
        """Format pseudo-labels for the built-in DetectionLoss"""
        batch = {
            'batch_idx': pseudo_labels[:, 0].long(),
            'cls': pseudo_labels[:, 1].long(),
            'bboxes': pseudo_labels[:, 2:6]  # xywh format
        }
        return batch
    
    def _validate(self):
        """Validate the teacher model on validation data"""
        # Set teacher model to evaluation mode
        self.teacher_model.model.eval()
        
        # Load the data.yaml configuration
        with open(self.data_yaml, errors='ignore') as f:
            data_dict = yaml.safe_load(f)
        
        # Get validation data path
        val_path = data_dict.get('val', '')
        if not val_path:
            print("Warning: No validation path found, using training path")
            val_path = data_dict.get('train', '')
        
        # Run validation using Ultralytics built-in method
        results = self.teacher_model.val(
            data=self.data_yaml,
            batch=self.batch_size,
            imgsz=self.img_size,
            verbose=False
        )
        
        # Extract metrics
        metrics = {}
        if hasattr(results, 'results_dict'):
            metrics_dict = results.results_dict
            metrics = {
                'metrics/precision': metrics_dict.get('metrics/precision(B)', 0),
                'metrics/recall': metrics_dict.get('metrics/recall(B)', 0),
                'metrics/mAP50': metrics_dict.get('metrics/mAP50(B)', 0),
                'metrics/mAP50-95': metrics_dict.get('metrics/mAP50-95(B)', 0)
            }
        else:
            # Fallback if results format is different
            print("Warning: Could not extract standard metrics format")
            if hasattr(results, 'box'):
                metrics = {
                    'metrics/precision': getattr(results.box, 'precision', 0),
                    'metrics/recall': getattr(results.box, 'recall', 0),
                    'metrics/mAP50': getattr(results.box, 'map50', 0),
                    'metrics/mAP50-95': getattr(results.box, 'map', 0)
                }
        
        return metrics
    
    def _save_results(self):
        """Save training results and plots"""
        import json
        import matplotlib.pyplot as plt
        
        # Save metrics history
        with open(self.save_dir / 'metrics_history.json', 'w') as f:
            json.dump(self.metrics_history, f)
        
        # Plot metrics
        if self.metrics_history:
            epochs = list(range(1, len(self.metrics_history) + 1))
            
            # Create plot for mAP
            plt.figure(figsize=(10, 6))
            if all('metrics/mAP50' in m for m in self.metrics_history):
                map50 = [m.get('metrics/mAP50', 0) for m in self.metrics_history]
                plt.plot(epochs, map50, 'b-', label='mAP@0.5')
            
            if all('metrics/mAP50-95' in m for m in self.metrics_history):
                map50_95 = [m.get('metrics/mAP50-95', 0) for m in self.metrics_history]
                plt.plot(epochs, map50_95, 'r-', label='mAP@0.5:0.95')
            
            plt.xlabel('Epoch')
            plt.ylabel('mAP')
            plt.title('Mean Average Precision')
            plt.legend()
            plt.grid(True)
            plt.savefig(self.save_dir / 'map_metrics.png')
            
            # Create plot for precision/recall
            plt.figure(figsize=(10, 6))
            if all('metrics/precision' in m for m in self.metrics_history):
                precision = [m.get('metrics/precision', 0) for m in self.metrics_history]
                plt.plot(epochs, precision, 'g-', label='Precision')
            
            if all('metrics/recall' in m for m in self.metrics_history):
                recall = [m.get('metrics/recall', 0) for m in self.metrics_history]
                plt.plot(epochs, recall, 'm-', label='Recall')
            
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.title('Precision and Recall')
            plt.legend()
            plt.grid(True)
            plt.savefig(self.save_dir / 'precision_recall.png')
            
            plt.close('all')