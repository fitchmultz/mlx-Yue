"""Upload converted MLX weights and VAE to Hugging Face Hub."""
import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, login


def main():
    parser = argparse.ArgumentParser(description="Upload mlx-Yue models to Hugging Face")
    parser.add_argument("--repo-id", default="vanch007/mlx-Yue2-3B", help="Target Hugging Face repo ID")
    parser.add_argument("--folder", default="models/mlx-Yue2-3B", type=Path, help="Folder containing generator weights and README")
    parser.add_argument("--vae-folder", default="models/YuE2-Vae", type=Path, help="Folder containing VAE weights")
    parser.add_argument("--token", default=None, help="Hugging Face write token (defaults to HF_TOKEN env var)")
    parser.add_argument("--private", action="store_true", help="Create as private repository")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=True)

    api = HfApi(token=token)
    try:
        user = api.whoami()
        print(f"Logged in as Hugging Face user: {user.get('name', user.get('user', 'unknown'))}")
    except Exception as e:
        print(f"Error: Unable to authenticate with Hugging Face: {e}", file=sys.stderr)
        print("Please provide a valid token with write access via --token or HF_TOKEN.", file=sys.stderr)
        sys.exit(1)

    print(f"Creating/verifying repository: {args.repo_id}...")
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True, private=args.private)

    # 1. Upload generator files
    print(f"Uploading generator files from {args.folder} to {args.repo_id}...")
    api.upload_folder(
        folder_path=str(args.folder),
        repo_id=args.repo_id,
        repo_type="model",
        ignore_patterns=["vae/**", "vae"],
        commit_message="Update mlx-Yue2-3B generator weights and documentation",
    )

    # 2. Upload VAE weights to vae/ subfolder
    if args.vae_folder and args.vae_folder.is_dir():
        print(f"Uploading VAE files from {args.vae_folder} to {args.repo_id}/vae...")
        api.upload_folder(
            folder_path=str(args.vae_folder),
            repo_id=args.repo_id,
            path_in_repo="vae",
            repo_type="model",
            commit_message="Add native MLX 48kHz FP32 Oobleck VAE decoder weights",
        )

    print(f"Upload complete! Model URL: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
