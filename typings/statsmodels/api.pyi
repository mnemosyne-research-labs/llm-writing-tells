import numpy as np
from numpy.typing import ArrayLike, NDArray

from statsmodels.genmod.cov_struct import CovStruct

class _Families:
    class Family: ...
    class Gaussian(Family):
        def __init__(self) -> None: ...
    class Poisson(Family):
        def __init__(self) -> None: ...
    class NegativeBinomial(Family):
        def __init__(self, alpha: float = ...) -> None: ...

families = _Families

class GEEResults:
    params: NDArray[np.float64]
    converged: bool
    def cov_params(self) -> NDArray[np.float64]: ...

class GEE:
    def __init__(
        self,
        endog: ArrayLike,
        exog: ArrayLike,
        groups: ArrayLike,
        family: _Families.Family | None = ...,
        cov_struct: CovStruct | None = ...,
        offset: ArrayLike | None = ...,
    ) -> None: ...
    def fit(self, maxiter: int = ...) -> GEEResults: ...
