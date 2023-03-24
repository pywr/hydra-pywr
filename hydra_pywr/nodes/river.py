"""
    Fix issue in hydra-pywr where it accept max_flow instead of mrf, so we
    have to add this to make it understand both
"""
from pywr.parameters import load_parameter
from pywr.domains import river

class RiverGauge(river.RiverGauge):
    """
        Overwrite the original rivergauge
    """
    def __init__(self, *args, **kwargs):
        """Initialise a new RiverGauge instance with 'max_flow' replaced with 'mrf'

        Parameters
        ----------
        mrf : float
            The minimum residual flow (MRF) at the gauge
        mrf_cost : float
            The cost of the route via the MRF
        cost : float
            The cost of the other (unconstrained) route
        """

        if kwargs.get('max_flow'):
            mrf = kwargs.pop('max_flow')
            try:
                kwargs['mrf'] = load_parameter(kwargs['model'], mrf)
            except KeyError:
                kwargs['mrf'] = mrf

        super().__init__(*args, **kwargs)
