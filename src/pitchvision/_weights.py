import os


def download_from_gdrive(file_id: str, destination: str) -> str:
    """Downloads a public Google Drive file to `destination` via gdown,
    skipping the download if it's already there - point `destination` at a
    path on Drive to avoid re-downloading every Colab session."""
    if os.path.exists(destination):
        return destination
    try:
        import gdown
    except ImportError as exc:
        raise ImportError("pip install gdown to download this pretrained checkpoint.") from exc
    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
    gdown.download(id=file_id, output=destination, quiet=False)
    return destination
