import argparse
from ultralytics import YOLO
from dyplom_mag.mean_teacher_training.model import SFYOLO

def main():
    parser = argparse.ArgumentParser(description="Train YOLO on a single mini-dataset.")
    parser.add_argument('--input_model_path', type=str, required=True,
                        help='Path to the input model weights (the current state).')
    parser.add_argument('--output_model_path', type=str, required=True,
                        help='Path to save the updated model weights (will overwrite input).')
    parser.add_argument('--is_sf_yolo', action='store_true', default=False,
                        help='Use SFYOLO model instead of YOLO')
    parser.add_argument('--training_kwargs', nargs='+', default=[],
                        help='Additional keyword arguments for model.train() in the format key=value')

    args = parser.parse_args()

    # Parse the kwargs from command line
    train_kwargs = {}
    for kwarg in args.training_kwargs:
        if '=' in kwarg:
            key, value = kwarg.split('=', 1)
            # Try to convert the value to int, float, or bool if possible
            try:
                value = eval(value)
            except:
                pass  # Keep as string if eval fails
            train_kwargs[key] = value

    if args.is_sf_yolo:
        model = SFYOLO(args.input_model_path)
    else:
        model = YOLO(args.input_model_path)

    model.train(**train_kwargs)
    model.save(args.output_model_path)

if __name__ == "__main__":
    main()