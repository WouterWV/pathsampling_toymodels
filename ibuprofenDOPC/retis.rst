Ibuprofen-DOPC
==============

Simulation
----------
task = ppretis
steps = 100000
interfaces = [-3, -2.4, -2.15, -1.98, -1.83, -1.66, -1.25, -0.8, -0.585, -0.449, -0.349, -0.261, -0.171, 0, 0.171, 0.261]
zero_left = -3.4
permeability = True
#restart = pyretis.restart

System
------
units = gromacs
dimensions=1
temperature=303

Box
---
periodic = [False]

Engine
------
class = Langevin
timestep = 0.02
gamma = 50
high_friction = False
seed = 0

TIS settings
------------
freq = 0.0
maxlength = 1000000
aimless = True
allowmaxlength = False
zero_momentum = False
rescale_energy = False
sigma_v =  -1
seed = 0

Initial-path
------------
method = load
#method = restart
#load_folder = load
#load_and_kick = True

RETIS settings
--------------
swapfreq = 0.1
relative_shoots = None
nullmoves = True
swapsimul = True

Particles
---------
position = {'input_file': 'initial.xyz'}
velocity = {'generate': 'maxwell',
            'momentum': False,
            'seed': 0}
mass = {'COM': 206.31, 'angle': 0.0093}
name = ['COM', 'angle']
ptype = [0]

Forcefield settings
-------------------
description = Interpolated 2D function

Potential
---------
class = Ibuprofen
module = pot_ibuprofen.py

Orderparameter
--------------
class = OrderV
module = order_2dvec.py

Output settings
---------------
trajectory-file = -1
order-file = -1
energy-file = -1
restart-file = 100
backup = 'append'

Analysis settings
-----------------
tau_ref_bin = [-3.4, -3]
