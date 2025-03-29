import os
import time
import copy
import torch
import numpy as np
from pathlib import Path

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.ops import non_max_suppression, xyxy2xywhn
from ultralytics.utils import LOGGER, TQDM, RANK, yaml_load, IterableSimpleNamespace
from torch import distributed as dist

from dyplom_mag.mean_teacher_3.detect import SFDetectionModel, DetectionModel
from ultralytics.nn.tasks import attempt_load_one_weight
from dyplom_mag.target_augment.enhance_style import get_style_images
from dyplom_mag.target_augment.enhance_vgg16 import enhance_vgg16
from ultralytics.models import yolo


ROOT = Path(__file__).parent
DEFAULT_CFG_PATH = ROOT / "default.yaml"
DEFAULT_CFG_DICT = yaml_load(DEFAULT_CFG_PATH)
for k, v in DEFAULT_CFG_DICT.items():
    if isinstance(v, str) and v.lower() == "none":
        DEFAULT_CFG_DICT[k] = None
DEFAULT_CFG_KEYS = DEFAULT_CFG_DICT.keys()
DEFAULT_CFG = IterableSimpleNamespace(**DEFAULT_CFG_DICT)


class WeightEMA(torch.optim.Optimizer):
    """Custom EMA optimizer for the teacher model"""
    def __init__(self, teacher_params, student_params, alpha=0.999):
        self.teacher_params = list(teacher_params)
        self.student_params = list(student_params)
        self.alpha = alpha
        
        if len(self.teacher_params) != len(self.student_params):
            raise ValueError(f"Teacher and student parameter lengths don't match: {len(self.teacher_params)} vs {len(self.student_params)}")
        
        # Package teacher params for optimizer
        param_groups = [{"params": self.teacher_params}]
        defaults = dict()
        super(WeightEMA, self).__init__(param_groups, defaults)

    def step(self):
        for teacher_param, student_param in zip(self.teacher_params, self.student_params):
            teacher_param.data.mul_(self.alpha).add_(student_param.data, alpha=1 - self.alpha)


class SFMeanTeacherTrainer(DetectionTrainer):
    """
    Source-Free YOLO Trainer implementing mean teacher approach with style transfer.
    
    This trainer uses a teacher-student architecture with AdaIN style transfer
    for domain adaptation in object detection.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize the SF Mean Teacher Trainer.
        
        Args:
            cfg: Configuration for training
            overrides: Overrides for configuration
            _callbacks: Custom callbacks
        """
        # Extract SF-YOLO specific parameters from overrides before initializing parent
        self.conf_thres = overrides.get('conf_thres', 0.25)
        self.iou_thres = overrides.get('iou_thres', 0.45)
        self.teacher_alpha = overrides.get('teacher_alpha', 0.999)
        self.ssm_alpha = overrides.get('ssm_alpha', 0.5)
        self.style_alpha = overrides.get('style_alpha', 0.2)
        self.max_gt_boxes = overrides.get('max_gt_boxes', 20)
        self.style_path = overrides.get('style_path', "")
        
        # Encoder and decoder paths for AdaIN style transfer
        base_dir = Path(__file__).parent.parent
        self.encoder_path = overrides.get('encoder_path', str(base_dir / "target_augment" / "pre_trained" / "vgg16_ori.pth"))
        self.decoder_path = overrides.get('decoder_path', str(base_dir / "target_augment" / "models" / "decoder.pth"))
        self.fc1_path = overrides.get('fc1_path', str(base_dir / "target_augment" / "models" / "fc1.pth"))
        self.fc2_path = overrides.get('fc2_path', str(base_dir / "target_augment" / "models" / "fc2.pth"))
        self.save_style_samples = overrides.get('save_style_samples', False)

        # Style transfer model
        self.style_transfer = None

        # Initialize teacher and student models (will be properly set in setup_model)
        self.teacher_model = None
        self.teacher_ema = None
        
        # Call parent initializer
        super().__init__(cfg, overrides, _callbacks)
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return an SF Detection model instead of standard Detection model"""
        model = SFDetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model
    
    def get_teacher_model(self, cfg=None, weights=None, verbose=True):
        """Return a standard Detection model with synchronized weights from the student model"""
        # Create teacher model
        teacher_model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        
        # If weights are provided, load them first
        if weights:
            teacher_model.load(weights)
            
        # If student model already exists as a module, copy its weights to teacher
        if isinstance(self.model, torch.nn.Module):
            # Get state dictionaries
            teacher_state_dict = teacher_model.state_dict()
            student_state_dict = self.model.state_dict()
            
            # Copy weights from student to teacher for matching keys
            for key in teacher_state_dict:
                if key in student_state_dict:
                    teacher_state_dict[key].copy_(student_state_dict[key])
        
        return teacher_model
    
    def setup_model(self):
        """Set up teacher and student models for mean teacher training"""
        self.setup_teacher_model()
        ckpt = super().setup_model()
        
        # Student model is already set up by parent class
        # Create teacher model as a copy of student model
        # self.teacher_model = copy.deepcopy(self.model)
        
        # Initialize style transfer
        self.init_style_transfer()
        
        return ckpt
    
    def setup_teacher_model(self):
        """Load/create/download model for any task."""
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            self.teacher_model = self.get_teacher_model(cfg=self.model.yaml)
            return

        cfg, weights = self.model, None
        ckpt = None
        if str(self.model).endswith(".pt"):
            weights, ckpt = attempt_load_one_weight(self.model)
            cfg = weights.yaml
        elif isinstance(self.args.pretrained, (str, Path)):
            weights, _ = attempt_load_one_weight(self.args.pretrained)
        self.teacher_model = self.get_teacher_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)
        return ckpt
    
    def init_style_transfer(self):
        """Initialize the style transfer model for AdaIN"""
        # Create a class with attributes needed by the style transfer function
        class StyleOpt:
            def __init__(self, style_path, img_size, style_alpha, encoder_path, decoder_path, fc1_path, fc2_path, save_style_samples, device):
                self.style_path = style_path
                self.imgsz = img_size
                self.style_add_alpha = style_alpha
                self.random_style = style_path == ""
                self.cuda = device.type == "cuda"
                self.log_dir = "./enhance_style_samples"
                self.encoder_path = encoder_path
                self.decoder_path = decoder_path
                self.fc1 = fc1_path
                self.fc2 = fc2_path
                self.save_style_samples = save_style_samples
                self.imgs_paths = []  # This will be updated during training
                
        opt = StyleOpt(
            self.style_path,
            self.args.imgsz,
            self.style_alpha,
            self.encoder_path,
            self.decoder_path,
            self.fc1_path,
            self.fc2_path,
            self.save_style_samples,
            self.device
        )
        self.style_transfer = enhance_vgg16(opt)
    
    def _setup_train(self, world_size):
        """Set up training, extending parent method for mean teacher training"""
        # Call parent method to set up base training components
        super()._setup_train(world_size)
        
        # Set up teacher optimizer (EMA)
        teacher_params = list(self.teacher_model.parameters())
        student_params = list(self.model.parameters())
        
        # Create EMA optimizer for teacher
        self.teacher_optimizer = WeightEMA(
            teacher_params, 
            student_params, 
            alpha=self.teacher_alpha
        )
        
        # Place teacher model on device
        self.teacher_model = self.teacher_model.to(self.device)
        
        # Log setup information
        LOGGER.info("Source-Free Mean Teacher training with:")
        LOGGER.info(f"  - Teacher alpha: {self.teacher_alpha}")
        LOGGER.info(f"  - SSM alpha: {self.ssm_alpha}")
        LOGGER.info(f"  - Style alpha: {self.style_alpha}")
        LOGGER.info(f"  - Confidence threshold: {self.conf_thres}")
        LOGGER.info(f"  - IoU threshold: {self.iou_thres}")
    
    def preprocess_batch(self, batch):
        """
        Preprocess a batch for training, adding style-transferred images
        
        Args:
            batch (Dict): Dictionary containing batch data
            
        Returns:
            (Dict): Processed batch with additional style information
        """
        # Standard preprocessing from parent
        batch = super().preprocess_batch(batch)
        
        # Store original images for teacher model
        batch["orig_img"] = batch["img"].clone()
        
        # Add image paths to style transfer for random style selection
        if self.style_transfer:
            self.style_transfer.args.imgs_paths = batch.get("im_file", [None] * len(batch["img"]))
        
        # Apply style transfer for student model if style_transfer is available
        if self.style_transfer:
            # To apply style transfer, we need the images in 0-255 range
            imgs_255 = batch["orig_img"].clone() * 255.0
            styled_imgs = get_style_images(imgs_255, adain=self.style_transfer) / 255.0
            batch["img"] = styled_imgs
        
        return batch
    
    def get_pseudo_labels(self, imgs, teacher_predictions):
        """
        Create pseudo-labels from teacher predictions
        
        Args:
            imgs (torch.Tensor): Input images
            teacher_predictions (List): NMS-filtered predictions from teacher model
            
        Returns:
            torch.Tensor: Formatted pseudo-labels for student training
        """
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
            pseudo_labels = [torch.zeros((0, 6), device=self.device)]
        
        # Concatenate all pseudo-labels
        return torch.cat(pseudo_labels, dim=0) if len(pseudo_labels[0]) > 0 else pseudo_labels[0]
    
    def _do_one_batch(self, batch):
        """
        Process one batch for mean teacher training
        
        Args:
            batch (Dict): Batch data
            
        Returns:
            (torch.Tensor): Loss value
        """
        # SSM weight transfer (after first epoch)
        if self.epoch > 0 and self.ssm_alpha > 0:
            student_state_dict = self.model.state_dict()
            teacher_state_dict = self.teacher_model.state_dict()
            
            for name, param in student_state_dict.items():
                if name in teacher_state_dict:
                    param.data.copy_((1.0 - self.ssm_alpha) * param.data + 
                                    self.ssm_alpha * teacher_state_dict[name].data)
        
        # Teacher forward pass to generate pseudo-labels (in eval mode, no grad)
        batch = self.preprocess_batch(batch)
        print(batch.keys())
        with torch.no_grad():
            self.teacher_model.eval()
            teacher_output = self.teacher_model(batch["orig_img"])
            
            # Apply NMS to teacher predictions for pseudo-labeling
            teacher_predictions = non_max_suppression(
                teacher_output[0] if isinstance(teacher_output, tuple) else teacher_output,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres,
                max_det=self.max_gt_boxes
            )
        
        # Get pseudo-labels from teacher predictions
        pseudo_labels = self.get_pseudo_labels(batch["orig_img"], teacher_predictions)
        
        # Student forward pass using styled images
        self.model.train()
        student_output = self.model(batch["img"])
        
        # Compute loss using pseudo-labels
        compute_loss = self.model.init_criterion()
        loss, loss_items = compute_loss(student_output, pseudo_labels)
        
        return loss, loss_items
    
    def _do_train_epoch(self, pbar, ni, epoch):
        """
        Train for one epoch
        
        Args:
            pbar: Progress bar
            ni: Number of iterations
            epoch: Current epoch
        """
        self.model.train()
        self.teacher_model.eval()  # Teacher is always in eval mode for inference
        
        for i, batch in pbar:
            # Run callbacks
            self.run_callbacks("on_train_batch_start")
            # Process batch and get loss
            loss, self.loss_items = self._do_one_batch(batch)
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Optimize - Gradient accumulation
            if ni - self.last_opt_step >= self.accumulate:
                self.optimizer_step()
                self.last_opt_step = ni
                
                # Update teacher with EMA of student
                self.teacher_optimizer.step()
                
            # Update metrics
            if RANK in {-1, 0}:
                self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                # Update progress bar description
                self.update_pbar_description(pbar, batch, epoch, ni)
                self.run_callbacks("on_batch_end")
            
            self.run_callbacks("on_train_batch_end")
    
    def update_pbar_description(self, pbar, batch, epoch, ni):
        """Update progress bar with training information"""
        mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"
        loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
        pbar.set_description(
            ("%11s" * 2 + "%11.4g" * (2 + loss_length))
            % (
                f"{epoch + 1}/{self.epochs}",
                mem,
                *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),
                batch["img"].shape[0],
                batch["img"].shape[-1],
            )
        )
        
        # Plot training samples with pseudo-labels
        if self.args.plots and ni in self.plot_idx:
            self.plot_training_samples(batch, ni)
    
    def _do_train(self, world_size=1):
        """
        Override _do_train to implement mean teacher training logic
        
        Args:
            world_size (int): Number of GPUs to use
        """
        if world_size > 1:
            self._setup_ddp(world_size)
        
        self._setup_train(world_size)
        
        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        self.last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        
        self.run_callbacks("on_train_start")
        LOGGER.info(f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
                   f"Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n"
                   f"Logging results to {self.save_dir}\n"
                   f"Starting SF-YOLO mean teacher training for {self.epochs} epochs...")
        
        epoch = self.start_epoch
        self.optimizer.zero_grad()
        
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            self.scheduler.step()
            
            self.model.train()
            self.teacher_model.eval()
            
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            
            pbar = enumerate(self.train_loader)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()
            
            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            
            self.tloss = None
            ni = 0  # number of iterations
            
            # Train for one epoch
            self._do_train_epoch(pbar, ni, epoch)
            
            # Update learning rate
            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}
            self.run_callbacks("on_train_epoch_end")
            
            # Validation and model saving
            if RANK in {-1, 0}:
                # Validation
                if self.args.val or (epoch + 1 >= self.epochs):
                    self.metrics, self.fitness = self.validate()

                # Save metrics from teacher model
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                
                # Check if training should stop
                self.stop = self.stopper(epoch + 1, self.fitness) or (epoch + 1 >= self.epochs)
                
                # Save teacher and student models
                if self.args.save or (epoch + 1 >= self.epochs):
                    self.save_teacher_student()
            
            # Early stopping across all DDP ranks
            if RANK != -1:
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)
                self.stop = broadcast_list[0]
            
            if self.stop:
                break
            
            epoch += 1
        
        # Final validation with teacher model
        if RANK in {-1, 0}:
            LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {(time.time() - self.train_time_start) / 3600:.3f} hours.")
            # Use teacher for final validation
            self.model = self.teacher_model
            self.final_eval()
            # Plot metrics
            if self.args.plots:
                self.plot_metrics()
            
        self.run_callbacks("on_train_end")
    
    def save_teacher_student(self):
        """Save both teacher and student models"""
        # First save the standard "last" model (student)
        super().save_model()
        
        # Now save teacher model separately
        teacher_path = self.wdir / "last_teacher.pt"
        student_path = self.wdir / "last_student.pt"
        
        # Save teacher model
        ckpt = {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "model": self.teacher_model,
            "ema": None,  # No EMA for teacher
            "updates": 0,
            "optimizer": None,  # Don't save optimizer state for teacher
            "train_args": vars(self.args),
            "date": time.strftime("%Y-%m-%d-%H-%M-%S"),
            "version": None,
        }
        torch.save(ckpt, teacher_path)
        
        # Save student model separately
        ckpt = {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "model": self.model,
            "ema": self.ema.ema if self.ema else None,
            "updates": self.ema.updates if self.ema else 0,
            "optimizer": self.optimizer.state_dict(),
            "train_args": vars(self.args),
            "date": time.strftime("%Y-%m-%d-%H-%M-%S"),
            "version": None,
        }
        torch.save(ckpt, student_path)
        
        # If this is the best model, also save as best teacher
        if self.best_fitness == self.fitness:
            teacher_best = self.wdir / "best_teacher.pt"
            student_best = self.wdir / "best_student.pt"
            torch.save(ckpt, student_best)
            
            # Update teacher ckpt with best fitness
            ckpt["model"] = self.teacher_model
            ckpt["ema"] = None
            torch.save(ckpt, teacher_best)

    def get_validator(self):
        """Return a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        args = copy.copy(self.args)
        delattr(args, "conf_thres")
        delattr(args, "style_alpha")
        delattr(args, "teacher_alpha")
        delattr(args, "max_gt_boxes")
        delattr(args, "style_path")
        delattr(args, "iou_thres")
        delattr(args, "save_style_samples")
        delattr(args, "ssm_alpha")
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=args, _callbacks=self.callbacks
        )

    def validate(self):
        """
        Runs validation on test set using self.validator.

        The returned dict is expected to contain "fitness" key.
        """
        self.model, orig_model = self.teacher_model, self.model
        self.model.init_criterion()
        print(type(self.model), type(self.model.loss), type(self.model.criterion))
        res = super().validate()
        self.model = orig_model
        return res