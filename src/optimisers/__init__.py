from .optimisers_file import OptimiserBase, LineSearchOptimiser, \
    TrustRegionOptimiser, OptimType, ScipyOptimiser, \
    OptimHessianType
#OptimHessianType, OptimType,
# LineSearchOptimiser
from . import line_search
from . import hessian_approx

from . import constants