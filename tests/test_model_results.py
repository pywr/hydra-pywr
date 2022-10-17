import numpy as np
import pandas
import pytest

from pywr.model import Model
from pywr.recorders import (
    NumpyArrayNodeRecorder,
    NumpyArrayStorageRecorder
)

from hydra_pywr.nodes.hydropower import *
from hydra_pywr.runner import PywrFileRunner
from fixtures import (
    model_directory,
    hydropower_verification_model
)


class TestResults():
    def test_hydropower_results(self, hydropower_verification_model):
        runner = PywrFileRunner()
        runner.load_pywr_model_from_file(hydropower_verification_model)
        gerd_rec = NumpyArrayStorageRecorder(runner.model, runner.model.nodes["GERD"])
        runner.run_pywr_model()

        gerd_st_df = gerd_rec.to_dataframe()
        gerd_turbine_df = runner.model.recorders["hydropowerrecorder.GERD_turbine"].to_dataframe()

        """
          Verified values for start and end sequences of results period
        """
        """ 1. GERD Storage """
        gerd_st_start = np.array([[46360.63694919],
                                  [42965.88915086],
                                  [39208.38512899],
                                  [35785.00239353],
                                  [32702.58927283]])

        gerd_st_end = np.array([[40747.57945841],
                                [49263.9095238 ],
                                [53011.73528208],
                                [53127.665874  ],
                                [50834.14051615]])

        """ 2. GERD Hydropower """
        gerd_turbine_start = np.array([[1730.1752206 ],
                                       [1595.41837885],
                                       [1466.1889006 ],
                                       [1329.45736668],
                                       [1210.65347216]])

        gerd_turbine_end = np.array([[1070.2307505 ],
                                     [1384.66567242],
                                     [1706.67934078],
                                     [1827.94504144],
                                     [1831.7610369 ]])

        assert gerd_st_df.shape == (240, 1)
        assert gerd_turbine_df.shape == (240, 1)

        assert np.all(np.isclose(gerd_st_df[0:5].values, gerd_st_start))
        assert np.all(np.isclose(gerd_st_df[-5:].values, gerd_st_end))
        assert np.all(np.isclose(gerd_turbine_df[0:5].values, gerd_turbine_start))
        assert np.all(np.isclose(gerd_turbine_df[-5:].values, gerd_turbine_end))
