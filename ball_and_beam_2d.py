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
        
        # Default parameters for the ball and beam -- should match
        # ball_and_beam.sdf, where applicable.
        # Probably don't need all of these.
        self.g = 9.8
        self.m = 0.1 # kg
        self.R = 0.5 # m
        self.J_b = self.m*self.R**2#2.25*10**(-5) # kg m^2
        self.theta_bar = 0.5 # rad, < pi/2
        self.M = 0.2 # kg
        self.L = 3 # m
        self.J = self.L**2 #0.36 # kg m^
        
        self.beta = 0.2 # Nm/s
        self.b = self.beta/(self.m + (self.J_b/(self.R**3)))
        self.d = self.m/(self.m + (self.J_b/(self.R**3)))
        self.n = (self.m*self.g)/(self.m + (self.J_b/(self.R**3)))
        self.k_p = 250
        self.k1_bar = 1/self.n
        self.k_d = 50
        self.i = complex(0,1)

    ''' 
    The following two functions are from the paper that the controller 
    comes from and are used to calculate the control input
    '''
    def phi_c(self,r):
        return -2*np.imag(erf((self.i + 2*self.d*r)/(2*sqrt(self.d))))
    
    def phi_s(self,r):
        return 2*np.real(erf((self.i + 2*self.d*r)/(2*sqrt(self.d)))) 
        
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
    
        # Restrict the beam angle and ball position to be within the admissible set
        theta = np.clip(theta, -self.theta_bar, self.theta_bar)
        r = np.clip(r, -self.L, self.L)
            
        # Controller
        alpha_1 = (sqrt(pi)*exp(-1/(4*self.d)))/(4*sqrt(self.d))
        alpha_0 = 2*np.imag(erf(self.i/(2*sqrt(self.d))))
        Isinr = alpha_1*(alpha_0 + self.phi_c(r))
        Icosr = alpha_1*(self.phi_s(r))
        dVp_dtheta = self.n*self.k1_bar*(-sin(theta - r)*Isinr + cos(theta - r)*Icosr) - self.k_p*(r - theta)
        controller = -self.k_d*(-r_dot + theta_dot) - (self.n*sin(theta) + dVp_dtheta) - (-self.d*r*theta_dot**2 + self.d*self.k1_bar*r*r_dot*(r_dot + theta_dot)*exp(-self.d*r**2))
        y[:] = controller

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