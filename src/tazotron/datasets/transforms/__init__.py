"""Transforms for dataset preprocessing/augmentation."""

# Реэкспорт трансформов для удобного импорта.
from tazotron.datasets.transforms.crop import BilateralHipROICrop
from tazotron.datasets.transforms.femoral_head import AddFemoralHeadMasks
from tazotron.datasets.transforms.necro import AddRandomNecrosis
from tazotron.datasets.transforms.xray import RenderDRR

# Публичный API модуля трансформов.
__all__ = ["AddFemoralHeadMasks", "AddRandomNecrosis", "BilateralHipROICrop", "RenderDRR"]
