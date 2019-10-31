#
# Created by Lukas Lüftinger on 05/02/2019.
#
from .clf.svm import TrexSVM
from .clf.xgbm import TrexXGB


__all__ = ['TrexXGB', 'TrexSVM']
