# -*- coding: utf-8 -*-
# Copyright (c) 2022, PyRETIS Development Team.
# Distributed under the LGPLv2.1+ License. See LICENSE for more info.
"""The order parameter for the hysteresis example."""
import logging
from pyretis.orderparameter.orderparameter import OrderParameter
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.addHandler(logging.NullHandler())


class OrderX(OrderParameter):
    """A positional order parameter. lambda = x of the only particle
    """

    def __init__(self):
        """Initialise the order parameter.

        """
        txt = '"A positional order parameter. lambda = x of the only particle'
        super().__init__(description=txt)


    def calculate(self, system):
        """Calculate the order parameter.

        Parameters
        ----------
        system : object like :py:class:`.System`
            This object is used for the actual calculation, typically
            only `system.particles.pos` and/or `system.particles.vel`
            will be used. In some cases `system.forcefield` can also be
            used to include specific energies for the order parameter.

        Returns
        -------
        out : float
            The order parameter.

        """
        pos = system.particles.pos[0]
        lmb = pos[1]
        ortho = pos[0]
        return [lmb,ortho]
