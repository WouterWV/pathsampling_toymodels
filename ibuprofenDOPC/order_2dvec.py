import logging
import numpy as np
from pyretis.orderparameter.orderparameter import OrderParameter
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.addHandler(logging.NullHandler())


class OrderV(OrderParameter):
    """A positional order parameter. 
    Lambda = projection along unit vector V.
    """

    def __init__(self, Vx=1., Vy=0.):
        """Initialise the order parameter.
        """
        txt = 'A positional order parameter. Lambda = projected along'
        txt+=' unit vector V. Default = (1,0)'
        super().__init__(description=txt)
        V = np.array([Vx, Vy])
        self.V = V/np.linalg.norm(V)
        self.Vorth = np.array([-1.*(self.V)[1],(self.V)[0]])

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
        # OPTION 1: just 1 particle, that is 2-dimensional
        # x is first coord, theta is second coord
        # pos = system.particles.pos[0]     # select 1st particle
        # return [np.dot(pos, self.V),np.dot(pos,self.Vorth)]

        # OPTION 2: 2 particles, that are each 1-dimensional
        # particle 1 is x, particle 2 is theta
        pos = system.particles.pos.ravel()
        # print(pos.shape, self.V.shape, self.Vorth.shape)
        # print(ah)
        return [np.sum(pos*self.V),np.sum(pos*self.Vorth)]