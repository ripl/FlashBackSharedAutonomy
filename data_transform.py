"""Optional UR5 data preprocessing helper (set paths for your machine)."""
import argparse
import os

from diffusha.data_collection.buffer import UR5_DataPreprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dirs",
        nargs="+",
        required=True,
        help="One or more directories of raw UR5 trajectories",
    )
    parser.add_argument(
        "--save-dir",
        required=True,
        help="Output directory for processed transitions",
    )
    parser.add_argument("--new-name", default="ur5_processed")
    parser.add_argument("--remove-start", type=int, default=0)
    parser.add_argument("--remove-end", type=int, default=1)
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    UR5_DataPreprocess.ur5_data_extract(
        list_file_path=args.raw_dirs,
        save_dir=args.save_dir,
        new_name=args.new_name,
        remove_start=args.remove_start,
        remove_end=args.remove_end,
    )


if __name__ == "__main__":
    main()
