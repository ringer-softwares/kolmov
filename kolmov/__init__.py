__all__ = []

# Check if root is installed
has_root=False
try:
    import ROOT
    has_root=True
except:
    print('ROOT not installed at your system. correction table class not available.')


from . import utils
__all__.extend( utils.__all__ )
from .utils import *

if has_root:
    from . import crossval_table
    __all__.extend( crossval_table.__all__ )
    from .crossval_table import *

    from . import fit_table
    __all__.extend( fit_table.__all__ )
    from .fit_table import *
else:
    print('fit_table and crossval_table not available becouse ROOT is not installed at your system.')




