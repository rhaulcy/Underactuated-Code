import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML
from underactuated import PlanarRigidBodyVisualizer

from pydrake.all import (DirectCollocation, FloatingBaseType,
                         PiecewisePolynomial, RigidBodyTree, RigidBodyPlant,
                         SolutionResult)
from underactuated import (PlanarRigidBodyVisualizer)

from pydrake.all import (DiagramBuilder, FloatingBaseType, Simulator, VectorSystem,
                        ConstantVectorSource, SignalLogger, CompliantMaterial,
                         AddModelInstancesFromSdfString)
from IPython.display import HTML
from scipy.special import erf
from numpy import sqrt, sin, cos, pi
from math import exp
import matplotlib.pyplot as plt

from pydrake.all import (VectorSystem)

class BallAndBeam2dController(VectorSystem):
    def __init__(self, ballandbeam):
        '''
        Controls a ball and beam system described
        in ball_and_beam.sdf.

        :param ballandbeam: A pydrake RigidBodyTree() loaded
            from ball_and_beam.sdf.
        '''
        VectorSystem.__init__(self,
            4, # 4 inputs: r, theta, and their derivatives 
            1) # 1 output: The torque of the beam
        self.ballandbeam = ballandbeam
        
    def _DoCalcVectorOutput(self, context, u, x, y):
        '''
        Given the state of the ball and beam (as the input to this system,
        u), populates (in-place) the control input to the ball and beam
        (y). This is given the state of this controller in x, but
        this controller has no state, so x is empty.

        :param u: numpy array, length 4, full state of the ball and beam.
        :param x: numpy array, length 0, full state of this controller.
        :output y: numpy array, length 1, control input to pass to the
            ball and beam.
        '''
        
        theta, r, theta_dot, r_dot = u[0:4]
        
        # Working PD Controller
        # Outer-loop error:
        # Regulate r and rdot to 0
        # This is intentionally overdamped
        # so the ball doesn't roll too quickly,
        # to protect against oscillations around
        # the fixed point.
        err = (0 - r) + (0 - r_dot)
        
        # Inner loop controller: We know the dynamics
        # of r depend on theta (i.e. the ball rolls down
        # the beam), so choose a desired beam angle based
        # on the error of the ball position, and control
        # to it extremely aggressively with a PD controller
        theta_des = max(min(-err * 0.1, 0.5), -0.5)
        control_torque = 250*(theta_des - theta) - 50*(theta_dot)
        y[0] = control_torque

'''
Simulates a 2d ball and beam system from initial conditions x0 (which
should be a 4x1 np array) for duration seconds.
'''
def Simulate2dBallAndBeam(x0, duration):
    builder = DiagramBuilder()

    # Load in the ball and beam from a description file.
    tree = RigidBodyTree()
    AddModelInstancesFromSdfString(
        open("ball_and_beam.sdf", 'r').read(),
        FloatingBaseType.kFixed,
        None, tree)
    
    # A RigidBodyPlant wraps a RigidBodyTree to allow
    # forward dynamical simulation. 
    plant = builder.AddSystem(RigidBodyPlant(tree))
    
    # Spawn a controller and hook it up
    controller = builder.AddSystem(
        BallAndBeam2dController(tree))
    builder.Connect(plant.get_output_port(0), controller.get_input_port(0))
    builder.Connect(controller.get_output_port(0), plant.get_input_port(0))
    
    # Create a logger to log at 30hz
    state_log = builder.AddSystem(SignalLogger(plant.get_num_states()))
    state_log._DeclarePeriodicPublish(0.0333, 0.0) # 30hz logging
    builder.Connect(plant.get_output_port(0), state_log.get_input_port(0))

    # Create a simulator
    diagram = builder.Build()
    simulator = Simulator(diagram)
    
    # Don't limit realtime rate for this sim, since we
    # produce a video to render it after simulating the whole thing.
    #simulator.set_target_realtime_rate(100.0) 
    simulator.set_publish_every_time_step(False)

    # Force the simulator to use a fixed-step integrator,
    # which is much faster for this stiff system. (Due to the
    # spring-model of collision, the default variable-timestep
    # integrator will take very short steps. I've chosen the step
    # size here to be fast while still being stable in most situations.)
    integrator = simulator.get_mutable_integrator()
    integrator.set_fixed_step_mode(True)
    integrator.set_maximum_step_size(0.001)

    # Set the initial state
    state = simulator.get_mutable_context().get_mutable_continuous_state_vector()
    state.SetFromVector(x0)

    # Simulate!
    simulator.StepTo(duration)

    return tree, controller, state_log