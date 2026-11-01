"""Closed-loop inference wrapper around a LeRobot ACT policy (CPU-friendly).

Deterministic by construction: ACT is deterministic at inference (no VAE sampling), and the simulator is
seeded, so evaluating the same checkpoint on the same seed list is reproducible bit-for-bit on one machine.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .dataset import cam_key


class ActPolicy:
    def __init__(self, checkpoint: str | Path, device: str = "cpu", n_action_steps: int | None = None):
        from lerobot.policies.act.modeling_act import ACTPolicy

        from lerobot.policies.factory import make_pre_post_processors

        self.device = torch.device(device)
        self.policy = ACTPolicy.from_pretrained(str(checkpoint))
        if n_action_steps is not None:
            self.policy.config.n_action_steps = n_action_steps
        self.policy.to(self.device)
        self.policy.eval()
        # Normalisation lives in the processor pipelines (saved next to the weights), not in the policy.
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config, pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
        self.cameras = [k.split(".")[-1] for k in self.policy.config.input_features if k.startswith("observation.images.")]
        self.reset()

    def reset(self) -> None:
        self.policy.reset()

    @torch.no_grad()
    def act(self, obs: dict) -> np.ndarray:
        batch = {
            "observation.state": torch.as_tensor(obs["state"], dtype=torch.float32, device=self.device)[None],
        }
        for cam in self.cameras:
            img = torch.as_tensor(obs["images"][cam], dtype=torch.float32, device=self.device) / 255.0
            batch[cam_key(cam)] = img.permute(2, 0, 1)[None]  # (1, 3, H, W)
        batch = self.preprocessor(batch)
        action = self.policy.select_action(batch)
        action = self.postprocessor(action)
        return action[0].cpu().numpy().astype(np.float32)


class RandomInitActPolicy(ActPolicy):
    """Untrained ACT with the dataset's feature spec — for plumbing tests only."""

    def __init__(self, dataset_root: str | Path, repo_id: str = "local/plumbing", device: str = "cpu"):
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy

        meta = LeRobotDatasetMetadata(repo_id, root=dataset_root)
        input_features, output_features = {}, {}
        for k, f in meta.features.items():
            if k.startswith("observation.images."):
                input_features[k] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, f["shape"][0], f["shape"][1]))
            elif k == "observation.state":
                input_features[k] = PolicyFeature(type=FeatureType.STATE, shape=tuple(f["shape"]))
            elif k == "action":
                output_features[k] = PolicyFeature(type=FeatureType.ACTION, shape=tuple(f["shape"]))
        cfg = ACTConfig(input_features=input_features, output_features=output_features, chunk_size=20, n_action_steps=10,
                        device=device)
        self.device = torch.device(device)
        self.policy = ACTPolicy(cfg)
        self.policy.to(self.device).eval()
        from lerobot.policies.factory import make_pre_post_processors

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            cfg, dataset_stats=meta.stats, preprocessor_overrides={"device_processor": {"device": device}})
        self.cameras = [k.split(".")[-1] for k in input_features if k.startswith("observation.images.")]
        self.reset()
