import os
import glob
import logging
from typing import Optional, Tuple
from PIL import Image
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


class VideoFrameDataset(Dataset):
    def __init__(self, root_dir: str, transform: Optional[transforms.Compose] = None,
                 img_size: Tuple[int, int] = (64, 64), fraction: float = 1.0):
        self.root_dir = root_dir
        self.frames = []
        self.fraction = fraction

        image_extensions = ['.png', '.jpg', '.jpeg']
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']

        # Load image files
        for ext in image_extensions:
            img_paths = glob.glob(os.path.join(root_dir, '**', f'*{ext}'), recursive=True)
            self.frames.extend(img_paths)

        # Load frames from video files
        for ext in video_extensions:
            video_paths = glob.glob(os.path.join(root_dir, '**', f'*{ext}'), recursive=True)
            for video_path in video_paths:
                self._extract_frames_from_video(video_path)

        # Define default transform if none provided
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])

        # logging.info(f"Loaded {len(self.frames)} frames from '{root_dir}'.")

    def _extract_frames_from_video(self, video_path: str) -> None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Failed to open video: {video_path}")
            return

        # We can take just a fraction of the frames
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_to_extract = int(frame_count * self.fraction)
        # logging.info(f"Extracting {num_to_extract}/{frame_count} frames from video: {video_path}")

        count = 0
        success, frame = cap.read()
        while success and count < num_to_extract:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame)
            self.frames.append(pil_image)
            success, frame = cap.read()
            count += 1

        cap.release()

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int):
        item = self.frames[idx]
        # Load image from file path if necessary
        image = Image.open(item).convert('RGB') if isinstance(item, str) else item
        return self.transform(image)


if __name__ == '__main__':
    dataset = VideoFrameDataset(root_dir='data')
    # logging.info(f"Number of frames loaded: {len(dataset)}")