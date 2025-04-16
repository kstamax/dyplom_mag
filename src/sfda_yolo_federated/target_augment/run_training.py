import argparse
from dyplom_mag.target_augment.train import train_style_transfer, TrainingConfig

def main(content_dir, style_dir, max_iter=2000):
    config = TrainingConfig(
        content_dir=content_dir,
        style_dir=style_dir,
        max_iter=max_iter,
        n_threads=1,
        batch_size=1,
    )
    train_style_transfer(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Client for SFYOLO")
    parser.add_argument("--content_dir", type=str, required=True, help="path to images")
    parser.add_argument("--style_dir", type=str, help="Path to style")
    parser.add_argument("--max_iter", type=int, default="2000",help="max iters")

    args = parser.parse_args()
    main(args.content_dir, args.style_dir, args.max_iter)