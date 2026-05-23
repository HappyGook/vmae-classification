from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder
import torchvision.transforms.functional as tf


class BusViolenceDataset(Dataset):

    LABEL_MAP = {
        "VIOLENCE":    1,
        "NONVIOLENCE": 0,
    }
    SUBDIR_MAP = {
        "VIOLENCE": "Violence",
        "NONVIOLENCE": "NoViolence",
    }

    def __init__(self, root_dir, processor, n_frames: int = 16, stride: int = 4):
        self.processor = processor
        self.n_frames  = n_frames
        self.stride    = stride

        self.samples: list[tuple[Path, int]] = []
        skipped = 0
        seen = set()

        for line in (root_dir / "train.txt").read_text().splitlines():
            filename = line.strip()
            if not filename or filename in seen:
                continue
            seen.add(filename)

            prefix = filename.upper().split("_")[0]  # "VIOLENCE" / "NONVIOLENCE"
            if prefix not in self.LABEL_MAP:
                skipped += 1
                continue

            path = root_dir / self.SUBDIR_MAP[prefix] / filename
            if not path.exists():
                print(f"[dataset] missing file: {path}")
                skipped += 1
                continue

            self.samples.append((path, self.LABEL_MAP[prefix]))

        if skipped:
            print(f"[dataset] skipped {skipped} entries (unknown prefix or missing file)")

        num_violence  = sum(1 for _, l in self.samples if l == 1)
        num_non_violence = sum(1 for _, l in self.samples if l == 0)
        print(f"[dataset] loaded {len(self.samples)} clips  "
              f"(violence={num_violence}, non-violence={num_non_violence})")


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        frames = self._load_and_sample(path)

        # squeeze the batch dim so DataLoader can re-batch correctly
        pixel_values = self.processor(
            frames, return_tensors="pt"
        ).pixel_values.squeeze(0)          # → (T, C, H, W)  e.g. (16, 3, 224, 224)

        return pixel_values, label, str(path.name)

    # ------------------------------------------------------------------
    def _load_and_sample(self, path: Path) -> list[Image.Image]:

        try:
            decoder = VideoDecoder(path)
            total   = len(decoder)

            indices = list(range(0, total, self.stride))[: self.n_frames]
            while len(indices) < self.n_frames:
                indices.append(indices[-1])

            batch = decoder.get_frames_at(indices=indices)
            # batch.data: Tensor (N, C, H, W)
            return [tf.to_pil_image(batch.data[i]) for i in range(len(indices))]

        except Exception as exc:
            print(f"[dataset] could not decode {path.name}: {exc}")
            return [Image.new("RGB", (224, 224), color=(0, 0, 0))] * self.n_frames