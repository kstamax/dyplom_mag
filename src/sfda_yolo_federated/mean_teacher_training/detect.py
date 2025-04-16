from ultralytics.nn.tasks import DetectionModel

from dyplom_mag.mean_teacher_training.sf_yolo_loss import SFYOLOv8Loss


class SFDetectionModel(DetectionModel):
    """Custom YOLO model for Source-Free learning that works with pseudo-labels"""

    def __init__(self, cfg=None, ch=3, nc=None, verbose=True):
        """Initialize the Source-Free YOLO model

        Args:
            cfg: Model configuration file or dictionary
            ch: Number of input channels
            nc: Number of classes
            verbose: Whether to print model information
        """
        super().__init__(cfg, ch, nc, verbose)

    def init_criterion(self):
        """Initialize the custom SF-YOLO loss criterion"""
        return SFYOLOv8Loss(self)
