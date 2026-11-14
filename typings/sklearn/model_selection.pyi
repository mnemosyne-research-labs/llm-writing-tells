from collections.abc import Iterator

import numpy as np
from numpy.typing import ArrayLike, NDArray

class GroupKFold:
    def __init__(self, n_splits: int = ...) -> None: ...
    def split(
        self, X: ArrayLike, y: ArrayLike, groups: ArrayLike
    ) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]: ...
