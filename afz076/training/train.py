"""Train one model on GTSRB.

Examples (run from the project root):
  python training/train.py --data-root /content/gtsrb --model resnet18 --weather-p 0   --name resnet18_clean
  python training/train.py --data-root /content/gtsrb --model resnet18 --weather-p 0.5 --name resnet18_weather
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import MODEL_NAMES  # noqa: E402
from src.data import find_root, group_split, load_arrays, load_index  # noqa: E402
from src.trainer import train_model  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, help="folder containing Train/ and Test/")
    p.add_argument("--model", choices=MODEL_NAMES, required=True)
    p.add_argument("--name", default="", help="run name (default: <model>_clean or <model>_weather)")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weather-p", type=float, default=0.0, help="probability of weather augmentation per image")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--out-root", default="outputs")
    p.add_argument("--limit", type=int, default=0, help="use only the first N training images (smoke tests)")
    args = p.parse_args()

    root = find_root(args.data_root)
    df = load_index(root, "train")
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=0).reset_index(drop=True)
    tr_df, val_df = group_split(df)
    print(f"train images: {len(tr_df)} | validation images: {len(val_df)} "
          f"| distinct signs in validation: {val_df['group'].nunique()}")

    tag = "" if args.limit else "_full"
    train_x, train_y = load_arrays(tr_df, cache=f"{args.cache_dir}/train{tag}.npz")
    val_x, val_y = load_arrays(val_df, cache=f"{args.cache_dir}/val{tag}.npz")

    name = args.name or f"{args.model}_{'weather' if args.weather_p > 0 else 'clean'}"
    train_model(train_x, train_y, val_x, val_y, args.model, f"{args.out_root}/{name}", epochs=args.epochs,
                batch_size=args.batch_size, lr=args.lr, weather_p=args.weather_p,
                pretrained=not args.no_pretrained)


if __name__ == "__main__":
    main()
