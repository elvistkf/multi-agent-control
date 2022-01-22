import numpy as np
import scipy.optimize
from .agents import BaseAgent

class EventTrigger():
    """Base class for event-triggers
    """
    def __init__(self):
        pass
    
    def trigger_function(self, agent: BaseAgent) -> bool:
        """Determine if the agent should trigger

        Args:
            agent (BaseAgent): agent in consideration

        Returns:
            bool: whether or not to trigger
        """
        return NotImplementedError

class RandomTrigger(EventTrigger):
    """Randomised triggering law
    """
    def __init__(self, p: float = 0.5):
        """Initialise the random trigger

        Args:
            p (float, optional): probability of triggering. Defaults to 0.5.
        """
        self.prob = p
        
    def trigger_function(self, agent: BaseAgent) -> int:
        return np.random.binomial(1, self.prob)

class DeterministicEventTrigger(EventTrigger):
    def __init__(self, beta = 0.1, offset = 0):
        self.beta = beta
        self.offset = offset

    def trigger_function(self, agent: BaseAgent) -> bool:
        e_norm = np.linalg.norm(agent.get_state() - agent.get_external_state())
        threshold = 0
        degree = agent.get_degree()
        for i in range(agent.num_neighbours):
            neighbour = agent.get_neighbour(i)
            weight = agent.get_weight(i)
            threshold += -weight * (np.linalg.norm(neighbour.get_external_state() - agent.get_external_state())) ** 2
        threshold *= self.beta / degree
        threshold += e_norm ** 2

        return threshold > self.offset

class StochasticEventTrigger(EventTrigger):
    def __init__(self, rand_min = 0.05, rand_dist = 'uniform', opt_lambda = 1, opt_q = 1, beta = 0.1, kappa = 1.05, decay: callable = lambda t: 1):
        self.kappa = kappa
        self.beta = beta
        self.decay = decay
        self.rand_min = rand_min
        self.opt_lambda = opt_lambda
        self.opt_q = opt_q
        self.rand_dist = rand_dist

        if self.rand_dist == 'optimal':
            a = self.rand_min
            mu_bar=0.25*(-(1+a)+np.sqrt(a**2 + 18*a + 1))
            self.mu_bar = mu_bar
            if 0 < self.opt_lambda < 1:
                q = self.opt_q
                Lstar = self.opt_lambda
                log_kappa = np.log(self.kappa)
                self.theta = np.power(self.opt_lambda, q)
                const = (1-Lstar)*log_kappa + Lstar*(a/(2*mu_bar*mu_bar)-np.log(mu_bar)+log_kappa-1.5) - log_kappa
                F = lambda x: self.theta*(1-x)*(x-a)/(2*(x**2)) - np.log(x) - const if x >= mu_bar else 10
                mu = scipy.optimize.broyden1(F, np.exp(-self.theta))
                self.rand_alpha = ((1-a)/self.theta-1)*((mu-a)/(1-a))
                self.rand_beta = ((1-a)/self.theta-1)*((1-mu)/(1-a))

    def distribution(self) -> float:
        a = self.rand_min
        
        if self.rand_dist == 'uniform':
            return np.random.uniform(a, 1)
        elif self.rand_dist == 'optimal':
            if self.opt_lambda == 0:
                return 1.0
            elif self.opt_lambda == 1:
                p = (self.mu_bar - a)/(1 - a)
                return (1-a) * np.random.binomial(1, p) + a
            else:
                return (1-a) * np.random.beta(a=self.rand_alpha, b=self.rand_beta) + a
        
    def trigger_function(self, agent: BaseAgent) -> bool:
        rand_var = self.distribution()
        e_norm = np.linalg.norm(agent.get_state() - agent.get_external_state())
        threshold = 0
        degree = agent.get_degree()
        for i in range(agent.num_neighbours):
            neighbour = agent.get_neighbour(i)
            weight = agent.get_weight(i)
            threshold += -weight * (np.linalg.norm(neighbour.get_external_state() - agent.get_external_state())) ** 2
        threshold *= self.beta / degree
        threshold += e_norm ** 2
        decay = self.decay(agent.get_time())
        
        if decay > 1e-30:
            return np.log(rand_var) > np.log(self.kappa) - degree * threshold / decay
        else:
            return threshold > 0
            
class SelfTrigger(EventTrigger):
    def __init__(self, decay_coefficient = 5, decay_rate = 5, decay_min = 0.0001):
        self.decay_rate = decay_rate
        self.decay_min = decay_min
        self.decay_coefficient = decay_coefficient

    def trigger_function(self, agent: BaseAgent) -> bool:
        e_norm = np.linalg.norm(agent.get_state() - agent.get_external_state())
        decay = self.decay_coefficient * np.exp(-self.decay_rate * agent.get_time()) + self.decay_min
        return e_norm ** 2 > decay

class StochasticSelfTrigger(EventTrigger):
    def __init__(self, rand_min = 0.05, kappa = 1.05, decay: callable = lambda t: 1):
        self.kappa = kappa
        self.decay = decay
        self.rand_min = rand_min

    def distribution(self) -> float:
        a = self.rand_min
        return np.random.uniform(a, 1)

    def trigger_function(self, agent: BaseAgent) -> bool:
        rand_var = self.distribution()
        e_norm = np.linalg.norm(agent.get_state() - agent.get_external_state())
        threshold = 0
        degree = agent.get_degree()
        for i in range(agent.num_neighbours):
            neighbour = agent.get_neighbour(i)
            weight = agent.get_weight(i)
            threshold += -weight * (np.linalg.norm(neighbour.get_external_state() - agent.get_external_state())) ** 2
        threshold *= self.beta / degree
        threshold += e_norm ** 2
        # decay = self.decay_max * np.exp(-self.decay_rate * agent.get_time()) + self.decay_min
        decay = self.decay(agent.get_time())
        
        if decay > 1e-30:
            return np.log(rand_var) > np.log(self.kappa) - degree * threshold / decay
        else:
            return threshold > 0

class DynamicEventTrigger(EventTrigger):
    def __init__(self, beta = 1.5, sigma = 0.4, delta = 0.5, chi0 = 0.5, theta = 2):
        self.beta = beta
        self.sigma = sigma
        self.delta = delta
        self.chi = chi0
        self.theta = theta

    def trigger_function(self, agent: BaseAgent) -> bool:
        e_norm = np.linalg.norm(agent.get_state() - agent.get_external_state())
        threshold = 0
        degree = agent.get_degree()
        for i in range(agent.num_neighbours):
            neighbour = agent.get_neighbour(i)
            weight = agent.get_weight(i)
            threshold += weight * (np.linalg.norm(neighbour.get_external_state() - agent.get_external_state())) ** 2
        threshold *= self.sigma / 4
        chi_dot = -self.beta * self.chi - self.delta * (degree * (e_norm ** 2) - threshold)
        self.chi += chi_dot * agent.get_step_size()

        return self.theta * (degree * (e_norm ** 2) - threshold) >= self.chi

class TimeTrigger(EventTrigger):
    """Periodic time-based triggering law
    """
    def __init__(self, period: int = 5):
        """Initialise the trigger

        Args:
            period (int, optional): period of triggering in terms of time steps. Defaults to 5.
        """
        self.period = period
        self.k = 0

    def trigger_function(self, agent: BaseAgent) -> bool:
        self.k += 1
        return self.k % self.period == 0
