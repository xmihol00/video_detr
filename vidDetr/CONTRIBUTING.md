# Contributing to VideoDETR

Thank you for your interest in contributing to VideoDETR! This document
provides guidelines and conventions for working with the codebase.

---

## Code Style

### Naming Conventions

VideoDETR uses **camelCase** for all user-authored code:

| Element | Convention | Example |
|---------|-----------|---------|
| Functions / methods | `camelCase` | `trainOneEpoch()`, `buildVideoDETR()` |
| Variables | `camelCase` | `numFrames`, `queriesPerFrame` |
| Classes | `PascalCase` | `VideoDETR`, `VideoSequenceDataset` |
| Constants | `UPPER_SNAKE_CASE` | `CACHE_VERSION`, `IMAGENET_MEAN` |
| CLI arguments | `--camelCase` | `--numFrames`, `--batchSize` |
| File names | `snake_case.py` | `video_detr.py`, `contrastive_loss.py` |

> **Exception**: The vendored DETR code in `models/detr/` and `util/`
> retains the original `snake_case` naming from Facebook's codebase.

### Imports

* All imports within the package use **absolute paths** rooted at
  `vidDetr`:
  ```python
  from vidDetr.util.misc import NestedTensor
  from vidDetr.models.detr.backbone import build_backbone
  ```
* Relative imports are used **only within the same sub-package**:
  ```python
  from .temporal_encoding import TemporalPositionEncoding
  ```
* Never use `sys.path` manipulation.

### Type Hints

Add type hints to all function signatures. Use `typing` imports for
complex types:

```python
from typing import Dict, List, Optional, Tuple

def myFunction(
    inputs: List[Tensor],
    targets: Optional[Dict[str, Tensor]] = None,
) -> Tuple[Tensor, Dict[str, float]]:
    ...
```

### Docstrings

Every public class and function should have a docstring. Use Google-style
or NumPy-style formatting:

```python
def myFunction(x: Tensor, threshold: float = 0.5) -> Tensor:
    """
    Brief one-line description.

    Longer explanation if needed.

    Args:
        x: Input tensor of shape [B, N, D].
        threshold: Confidence threshold (default: 0.5).

    Returns:
        Filtered tensor of shape [B, M, D] where M <= N.
    """
```

---

## Architecture Guidelines

### Adding New Components

1. **Self-contained modules**: Each new component should be a standalone
   `nn.Module` with a clear `forward()` signature.
2. **Builder pattern**: Provide a `build<Component>(args)` factory
   function that constructs the module from CLI arguments.
3. **Export from `__init__.py`**: Add your class and builder to the
   appropriate sub-package's `__init__.py`.
4. **Tests**: Add a test function in `test_video_detr.py` that verifies
   the component builds and runs a forward pass on CPU with synthetic
   data.

### Loss Functions

Follow the existing pattern in `video_criterion.py`:

```python
def lossMyLoss(self, outputs, targets, indices, numBoxes, **kwargs):
    # 1. Extract relevant predictions
    # 2. Build targets from indices
    # 3. Compute loss
    # 4. Return dict: {'loss_my_loss': tensor}
    ...
```

Register in `getLoss()` and add to `weightDict` in
`buildVideoCriterion()`.

### Datasets

Each dataset class must:
* Inherit from `torch.utils.data.Dataset`
* Return `(images, targets)` tuples from `__getitem__`
* Provide a collate function that produces `[N, B]` frame-first layout
* Include a `build<Dataset>(args)` factory

---

## Testing

Run the test suite before submitting changes:

```bash
python -m vidDetr.test_video_detr
```

All tests should pass on CPU without any dataset or GPU.

---

## Pull Request Checklist

- [ ] Code follows the naming conventions above
- [ ] All new functions have type hints and docstrings
- [ ] New components have corresponding tests in `test_video_detr.py`
- [ ] Imports use absolute `vidDetr.*` paths (no `sys.path` hacks)
- [ ] New CLI arguments are documented in the README
- [ ] The test suite passes: `python -m vidDetr.test_video_detr`
