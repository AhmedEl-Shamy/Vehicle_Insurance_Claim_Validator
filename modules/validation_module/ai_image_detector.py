import io
import random
from collections import OrderedDict
import numpy as np
import torch
import requests
from PIL import Image, ImageFile
import albumentations as A
import albumentations.pytorch as Ap

# Import architecture module from the PoliMi repository
# from utils import architectures
import architectures

ImageFile.LOAD_TRUNCATED_IMAGES = True


class AIImageDetector:
    def __init__(self, weights_path: str = "synth_vs_real.pth", M: int = 400, device: str = None):
        """
        Initializes the PoliMi Synthetic Image Detector using 800-patch aggregation.
        
        :param weights_path: Path to synth_vs_real.pth
        :param M: Number of top patches to aggregate (default 400)
        :param device: 'cuda', 'cpu', or auto-detected
        """
        self.M = M
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))

        # Load EfficientNet-B4 network architecture
        network_class = getattr(architectures, 'EfficientNetB4')
        net = network_class(n_classes=2, pretrained=False).eval().to(self.device)

        # Load weights and adjust keys if needed
        state_tmp = torch.load(weights_path, map_location='cpu')
        if 'net' not in state_tmp.keys():
            state = OrderedDict({'net': OrderedDict()})
            [state['net'].update({'model.{}'.format(k): v}) for k, v in state_tmp.items()]
        else:
            state = state_tmp

        net.load_state_dict(state['net'], strict=True)
        self.net = net

        # Setup patch normalization and crop transforms
        net_normalizer = self.net.get_normalizer()
        transform = [
            A.Normalize(mean=net_normalizer.mean, std=net_normalizer.std),
            Ap.transforms.ToTensorV2()
        ]
        self.trans = A.Compose(transform)
        self.cropper = A.RandomCrop(width=96, height=96, always_apply=True, p=1.0)

    def predict(self, img_array: np.ndarray) -> float:
        """
        Runs patch-based synthetic image detection on a NumPy RGB image array.
        Returns the aggregated score rounded to 3 decimal places.
        """
        # Fixed seeds match PoliMi CLI script for deterministic outputs
        random.seed(21)
        np.random.seed(21)
        torch.manual_seed(21)

        # Format check: Convert grayscale to RGB
        if img_array.ndim < 3:
            img2 = np.zeros((img_array.shape[0], img_array.shape[1], 3), dtype=np.uint8)
            img2[:, :, 0] = img_array
            img2[:, :, 1] = img_array
            img2[:, :, 2] = img_array
            img_array = img2.copy()

        # Format check: Omit alpha channel
        if img_array.shape[2] > 3:
            img_array = img_array[:, :, :3]

        # Upscale if smaller than 256x256
        if img_array.shape[0] < 256 or img_array.shape[1] < 256:
            img_array = A.SmallestMaxSize(max_size=256, interpolation=1, always_apply=True, p=1.0)(image=img_array)['image']

        # Extract 800 random 96x96 patches
        all_patches = [self.cropper(image=img_array)['image'] for _ in range(800)]

        # Normalize patches
        transf_patch_list = [self.trans(image=patch)['image'] for patch in all_patches]
        transf_patch_tensor = torch.stack(transf_patch_list, dim=0).to(self.device)

        # Run inference and aggregate top M patches
        with torch.no_grad():
            patch_scores = self.net(transf_patch_tensor)

        # Compute top-M average score
        raw_score = torch.mean(torch.sort(patch_scores[:, 1])[0][-self.M:]).item()
        return round(raw_score, 3)

    def predict_from_url(self, image_url: str) -> float:
        """
        Downloads an image directly from Supabase Storage or external URL and returns score.
        """
        response = requests.get(image_url, timeout=15)
        response.raise_for_status()

        # Convert PIL byte stream to RGB NumPy array
        pil_img = Image.open(io.BytesIO(response.content)).convert("RGB")
        img_array = np.array(pil_img)

        return self.predict(img_array)

    def predict_batch_from_urls(self, image_urls: list[str]) -> tuple[float, list[dict]]:
        """
        Evaluates multiple image URLs for a single claim.
        Returns max_score across all images and per-image details list.
        """
        image_results = []
        scores = []

        for url in image_urls:
            try:
                score = self.predict_from_url(url)
            except Exception as e:
                print(f"[AIImageDetector] Error processing URL {url}: {e}")
                score = 0.0

            image_results.append({"url": url, "score": score})
            scores.append(score)

        max_score = max(scores) if scores else 0.0
        return max_score, image_results