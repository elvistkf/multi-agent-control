import numpy as np
from numpy import random
from typing import Any, Callable, List

from .common import is_scalar, is_integer, length, calculate_gradient, calculate_hessian, calculate_convex_optimal
from .event_trigger import EventTrigger

class BaseAgent():
    """Base class for agents
    """
    def __init__(self, state_dim: int = 1, state_range: tuple = (None, None), init_state: List[float] = None, step_size: float = 1):
        """Initialise the agent object

        Args:
            state_dim (int, optional): state dimension. Defaults to 1.
            state_range (tuple, optional): range of states to be generated if initial state is not provided. Defaults to (None, None).
            init_state (List[float], optional): initial states. Defaults to None.
            step_size (float, optional): time discretisation size. Defaults to 1.

        Raises:
            TypeError: agent state dimension is not an integer
            ValueError: agent dimension is not positive
            ValueError: agent state does not match the state dimension
            TypeError: agent state range has invalid format
            ValueError: agent state range does not match the dimension
            ValueError: step size is not positive
        """
        if not is_integer(state_dim):
            raise TypeError('Agent state dimension is not interger')

        if state_dim <= 0:
            raise ValueError('Agent dimension is not positive')
        
        if init_state is not None:
            if length(init_state) != state_dim:
                raise ValueError('Agent state does not match the state dimension')

        if length(state_range) != 2:
            raise TypeError('Agent state range has invalid format. Expected a two-element tuple (low, high)')
        
        if any(length(i) != state_dim for i in state_range) and all(i is not None for i in state_range):
            raise ValueError('Agent state range does not match the state dimension')

        if step_size <= 0:
            raise ValueError('Step size is not positive')

        self.state_dim = state_dim
        self.state_range = state_range
        self.init_state = init_state
        self.step_size = step_size
        self.objective = None
        self.reset()

    def step(self):
        """Proceed with one step
        """
        self.dynamics()
        self.estimate()
        self.broadcast()
        self.log_stat()

    def set_triggering_law(self, trigger: 'EventTrigger'):
        """Set the event-triggering law

        Args:
            trigger (EventTrigger): event-triggering law to be set
        """
        self.triggering_law = trigger

    def set_objective_function(self, obj: Callable):
        """Set the objective function to be optimised if applicable

        Args:
            obj (Callable): objective function
        """
        self.objective = obj
        self.reset(opt=True)

    def estimate(self):
        """Estimate the agent state
        """
        return NotImplementedError

    def dynamics(self):
        """State dynamics of the agent
        """
        return NotImplementedError

    def set_control(self, u: List[float]):
        """Set the control input for the agent

        Args:
            u List[float]: control input
        """
        self.u = u
    
    def set_attribute(self, attr: str, value: Any):
        """Set an arbitrary attribute

        Args:
            attr (str): attribute name
            value (Any): attribute value
        """
        setattr(self, attr, value)
    
    def get_attribute(self, attr: str) -> Any:
        """Get an arbitrary attribute

        Args:
            attr (str): attribute name

        Returns:
            Any: attribute value
        """
        return getattr(self, attr)

    def broadcast(self):
        """Broadcast the current state estimation
        """
        try:
            self.gamma = int(self.triggering_law.trigger_function(self))
        except AttributeError:
            self.gamma = 1
        
        if self.gamma:
            self.z = np.copy(self.xh)
        

    def reset(self, opt = False):
        """Reset the agent

        Args:
            opt (bool, optional): whether or not the agent is solving an optimisation problem. Defaults to False.
        """
        if not opt:
            low, high = self.state_range if not any(i is None for i in self.state_range) else (-5*np.ones(self.state_dim), 5*np.ones(self.state_dim))
            self.x = self.init_state if self.init_state is not None else random.uniform(low, high)
        else:
            self.x = calculate_convex_optimal(self.objective, self.state_dim)
        self.x = self.x.astype(np.longfloat)
        self.y = np.copy(self.x)     # Measurement of state
        self.xh = np.copy(self.x)    # MMSE estimate of state
        self.z = np.copy(self.x)     # Last broadcasted state
        self.u = np.copy(self.x) * 0
        self.gamma = 1
        self.t = 0

        self.stat = dict(
            {
                't': [self.t],
                'x': [list(self.x)],
                'y': [list(self.y)],
                'xh': [list(self.xh)],
                'z': [list(self.z)],
                'u': [list(self.u)],
                'gamma': [self.gamma]
            }
        )
    
    def log_stat(self):
        """Log agent stats including time, states, measurement, etc.
        """
        self.t += self.step_size
        self.stat['t'].append(self.t)
        self.stat['x'].append(list(self.x))
        self.stat['y'].append(list(self.y))
        self.stat['xh'].append(list(self.xh))
        self.stat['z'].append(list(self.z))
        self.stat['u'].append(list(self.u))
        self.stat['gamma'].append(self.gamma)

    def set_neighbours(self, neighbours: 'List[BaseAgent]', weights: List[float]):
        """Set the neighbours of the agent in the communication network

        Args:
            neighbours (List[BaseAgent]): set of agents in the neighbourhood
            weights (List[float]): set of edge weights with the neighbours
        """
        self.neighbours = neighbours
        self.weights = weights
        self.num_neighbours = length(self.neighbours)
        self.degree = np.sum(self.weights)
    
    def add_neighbours(self, neighbour: 'BaseAgent', weight: float):
        """Add a new neighbour in the communication network

        Args:
            neighbour (BaseAgent): new neighbour agent to be added
            weight (float): edge weight with the new neighbour
        """
        try:
            self.neighbours.append(neighbour)
            self.weights.append(weight)
        except AttributeError:
            self.neighbours = [neighbour]
            self.weights = [weight]
        finally:
            self.num_neighbours = length(self.neighbours)
            self.degree = np.sum(self.weights)

    def get_neighbours(self) -> 'List[BaseAgent]':
        """Get the set of neighbouring agents

        Returns:
            List[BaseAgent]: set of agents in the neighbourhood
        """
        return self.neighbours

    def get_neighbour(self, index) -> 'BaseAgent':
        """Get the neighbouring agent with a specific index

        Returns:
            BaseAgnet: target agent in the neighbourhood
        """
        return self.neighbours[index]

    def get_weights(self) -> list:
        """Get the set of weights in the communication graph

        Returns:
            list: set of weights
        """
        return self.weights
    
    def get_weight(self, index: int) -> float:
        """Get the weight with an agent of specific index

        Args:
            index (int): target agent index

        Returns:
            float: weight
        """
        return self.weights[index]

    def get_degree(self) -> float:
        """Get the total degree

        Returns:
            float: degree
        """
        return self.degree

    def get_time(self) -> float:
        """Get the current time

        Returns:
            float: time
        """
        return self.t

    def get_state(self) -> List[float]:
        """Get the current state

        Returns:
            List[float]: current state
        """
        return self.x

    def get_measurement(self) -> List[float]:
        """Get the current measurement

        Returns:
            List[float]: current measurement
        """
        return self.y

    def get_estimate(self) -> List[float]:
        """Get the current state estimate

        Returns:
            List[float]: current state estimate
        """
        return self.xh

    def get_external_state(self) -> List[float]:
        """Get the current state perceived by the neighbours

        Returns:
            List[float]: current "external state"
        """
        return self.z

    def get_step_size(self) -> float:
        """Get the step size

        Returns:
            float: step size
        """
        return self.step_size

    def calculate_gradient(self) -> np.array:
        """Calculate the gradient at current state

        Raises:
            AttributeError: The agent does not have an objective function

        Returns:
            np.array: gradient 
        """
        if self.objective is None:
            raise AttributeError('The agent does not have an objective function')
        return calculate_gradient(self.objective, self.x)

    def calculate_hessian(self) -> np.array:
        """Calculate the hessian matrix at current state

        Raises:
            AttributeError: The agent does not have an objective function

        Returns:
            np.array: hessian matrix
        """
        if self.objective is None:
            raise AttributeError('The agent does not have an objective function')
        return calculate_hessian(self.objective, self.x)

class SingleIntegratorAgent(BaseAgent):
    """Single integrator agent with dynamics:
        
        dx/dt = u,
        y = x
    """
    def estimate(self):
        self.y = np.copy(self.x)
        self.xh = np.copy(self.y)

    def dynamics(self):
        """Dynamics of the single integrator agent

        Raises:
            ValueError: Control input u does not match the state dimension
        """
        if length(self.u) != self.state_dim:
            raise ValueError('Control input u does not match the state dimension')
        
        dt = self.step_size
        dx = self.u
        self.x += dx * dt

class BaseLinearAgent(BaseAgent):
    """Base class for continuous or discrete time LTI agent
    """
    def __init__(self, state_dim: int, A = None, B = None, C = None, state_range = (None, None), init_state = None, step_size = 1, obj_func = None):
        """Initialise the linear agent

        Args:
            state_dim (int): state dimension
            A ([type], optional): according to the dynamics above. Defaults to None.
            B ([type], optional): according to the dynamics above. Defaults to None.
            C ([type], optional): according to the dynamics above. Defaults to None.
            state_range (tuple, optional): range of states to be generated if initial state is not provided. Defaults to (None, None).
            init_state ([type], optional): initial states. Defaults to None.
            step_size (int, optional): step size. Defaults to 1.
            obj_func ([type], optional): objective function if applicable. Defaults to None.

        Raises:
            ValueError: Transition matrix A does not match the state dimension
            ValueError: Transition matrix A is not square
            ValueError: Transition matrix B does not match the state dimension
            ValueError: Transition matrix C does not match the state dimension
        """
        # TODO: extend the agent measurement model to y = Cx + Du
        if not is_scalar(A):
            A = np.array(A)
            shape = tuple(A.shape)
            if shape[1] != state_dim:
                raise ValueError('Transition matrix A does not match the state dimension')
            
            if shape[0] != shape[1]:
                raise ValueError('Transition matrix A is not square')
        
        if not is_scalar(B):
            B = np.array(B)
            if tuple(B.shape)[0] != state_dim:
                raise ValueError('Transition matrix B does not match the state dimension')

        if not is_scalar(C):
            C = np.array(C)
            if tuple(C.shape)[1] != state_dim:
                raise ValueError('Transition matrix C does not match the state dimension')
        
        super().__init__(state_dim, state_range, init_state, step_size, obj_func)
        self.A = A if A is not None else 1
        self.B = B if B is not None else 1
        self.C = C if C is not None else 1

    def estimate(self):
        self.y = np.matmul(self.C, self.x)
        self.xh = np.copy(self.y)

    def dynamics(self):
        try:
            if is_scalar(self.A):
                self.Ax = self.A * self.x
            else:
                self.Ax = np.matmul(self.A, self.x)
            
            if is_scalar(self.B):
                self.Bu = self.B * self.u
            else:
                self.Bu = np.matmul(self.B, self.u)

        except ValueError:
            raise ValueError('Control input u does not match the actuation matrix B')
        
        if length(self.Bu) != self.state_dim or self.Bu.shape[0] != self.state_dim:
            raise ValueError('Product of control input u and actuation matrix B does not match the state dimension')


class LinearContinuousAgent(BaseLinearAgent):
    """Linear continuous time agent
    """
    def dynamics(self):
        super().dynamics()
        dt = self.step_size
        dx = self.Ax + self.Bu
        self.x += dx * dt
    
class LinearDiscreteAgent(BaseLinearAgent):
    """Linear discrete time agent
    """
    def dynamics(self):
        super().dynamics()
        self.x = self.Ax + self.Bu

class NonlinearContinuousAgent(BaseAgent):
    """Nonlinear continuous time agent with dynamics:
    
        dx/dt = f(x,u,t),
            y = g(x,u,t)
    """
    def __init__(self, state_dim: int = 1, state_range = (None, None), init_state = None, step_size = 1, dynamics_func = None, estimate_func = None):
        self.dynamics_func = dynamics_func if dynamics_func is not None else lambda x,u,t: u
        self.estimate_func = estimate_func if estimate_func is not None else lambda x,u,t: x

        super().__init__(state_dim, state_range, init_state, step_size)
    
    def dynamics(self):
        self.x += self.dynamics_func(self.x, self.u, self.t) * self.step_size
        
    def estimate(self):
        self.y = self.estimate_func(self.x, self.u, self.t)
        self.xh = np.copy(self.y)
    
    def set_dynamics_function(self, dynamics_func: Callable):
        """Set the dynamics function f(x)

        Args:
            dynamics_func (Callable): dynamics function to be set
        """
        self.dynamics_func = dynamics_func
    
    def set_estimate_function(self, estimate_func: Callable):
        """Set the estimation function g(x)

        Args:
            estimate_func (Callable): estimation function to be set
        """
        self.estimate_func = estimate_func