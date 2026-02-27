# src/models/moirai/custom_concat_dataset_builder.py
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig

# import from the SAME tree as the rest of builders
from uni2ts.data.builder._base import (
    ConcatDatasetBuilder as _BaseConcat,
    DatasetBuilder as _BaseDB,
)

from uni2ts.data.builder.simple import SimpleFinetuneDatasetBuilder as UFin
from uni2ts.data.builder.simple import SimpleEvalDatasetBuilder as UEval

from torch.utils.data import ConcatDataset as _TorchConcat

class ConcatWithMeta(_TorchConcat):
    """torch ConcatDataset + metadata the trainer expects."""
    def __init__(self, datasets):
        super().__init__(datasets)
        child_num = [getattr(d, "num_ts", len(d)) for d in self.datasets]
        child_w   = [getattr(d, "dataset_weight", 1.0) for d in self.datasets]
        self.num_ts = sum(child_num)
        # preserve semantics: product equals sum of child products
        self.dataset_weight = (
            (sum(w * n for w, n in zip(child_w, child_num)) / self.num_ts)
            if self.num_ts > 0 else 1.0
        )

class CustomConcatDatasetBuilder(_BaseConcat):
    """Concat builder that accepts *args of child builders/configs and instantiates them."""

    def __init__(self, *builders):
        # Bypass _BaseConcat.__init__ (it contains the strict isinstance assert)
        _BaseDB.__init__(self)

        inst = []
        for b in builders:
            # turn Hydra nodes into objects
            if isinstance(b, (DictConfig, ListConfig)):
                b = instantiate(b)

            # last-resort: if a plain dict sneaks in, rebuild a builder
            if isinstance(b, dict):
                cls = UEval if "offset" in b else UFin
                b = cls(**b)

            inst.append(b)

        # Lightweight safety: they must look like builders
        for b in inst:
            if not hasattr(b, "load_dataset"):
                raise TypeError(f"{b!r} does not implement load_dataset(transform_map)")

        self.builders = tuple(inst)

    def load_dataset(self, transform_map):
        children = [b.load_dataset(transform_map) for b in self.builders]
        return ConcatWithMeta(children)