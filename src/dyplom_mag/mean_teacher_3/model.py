from ultralytics import YOLO
from dyplom_mag.mean_teacher_3.train import SFMeanTeacherTrainer

class SFYOLO(YOLO):
    """
    Source-Free YOLO (SF-YOLO) model for domain adaptation with mean teacher architecture.
    
    This class extends the standard YOLO class to support source-free domain adaptation 
    using mean teacher architecture with style transfer.
    """
    
    def __init__(self, model="yolo11n.pt", task=None, verbose=False):
        """
        Initialize SF-YOLO model.
        
        Args:
            model (str): Path to model file (.pt) or model name
            task (str, optional): Task type (detect, segment, classify, pose)
            verbose (bool): Display model info on load
        """
        super().__init__(model=model, task=task or "detect", verbose=verbose)
        
    @property
    def task_map(self):
        """Return the task map with SF-YOLO trainer."""
        task_map = super().task_map
        # Override the default detect trainer with our custom SF-YOLO trainer
        if "detect" in task_map:
            task_map["detect"]["trainer"] = SFMeanTeacherTrainer
        return task_map
    
    def train(self, 
              data=None,
              epochs=100,
              style_path="",
              teacher_alpha=0.999,
              ssm_alpha=0.5,
              style_alpha=0.2,
              conf_thres=0.25,
              iou_thres=0.45,
              max_gt_boxes=20,
              save_style_samples=False,
              **kwargs):
        """
        Train the SF-YOLO model using source-free domain adaptation.
        
        Args:
            data (str): Dataset config file path
            epochs (int): Number of training epochs
            style_path (str): Path to style images directory
            teacher_alpha (float): EMA coefficient for teacher model updates
            ssm_alpha (float): Weight for Stochastic Structure Matching
            style_alpha (float): Weight for style transfer
            conf_thres (float): Confidence threshold for pseudo-labels
            iou_thres (float): IoU threshold for NMS
            max_gt_boxes (int): Maximum number of ground truth boxes
            save_style_samples (bool): Whether to save style samples for debugging
            **kwargs: Additional arguments passed to the trainer
        
        Returns:
            (dict): Training metrics
        """
        # Set SF-YOLO specific parameters
        overrides = {
            "data": data,
            "epochs": epochs,
            "style_path": style_path,
            "teacher_alpha": teacher_alpha,
            "ssm_alpha": ssm_alpha,
            "style_alpha": style_alpha,
            "conf_thres": conf_thres,
            "iou_thres": iou_thres,
            "max_gt_boxes": max_gt_boxes,
            "save_style_samples": save_style_samples,
            **kwargs
        }
        
        # Call standard train method with our overrides
        return super().train(**overrides)