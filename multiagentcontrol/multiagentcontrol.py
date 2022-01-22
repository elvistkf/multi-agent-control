import numpy as np
import platform
import matplotlib as mpl
if platform.system() == 'Darwin':       # avoid bugs in some versions of matplotlib with macOS catalina
    mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import networkx as nx

from typing import Any, Callable, List, Union

from .agents import BaseAgent, SingleIntegratorAgent, LinearContinuousAgent, LinearDiscreteAgent
from .event_trigger import EventTrigger, DeterministicEventTrigger, StochasticEventTrigger, SelfTrigger
from .common import is_scalar, is_integer, length, calculate_convex_optimal, calculate_gradient, calculate_hessian

class MultiagentControl():
    def __init__(self, num_agents: int, id: int = None, graph: nx.Graph = None, agents: List[BaseAgent] = None, agent_type: BaseAgent = SingleIntegratorAgent, state_dim: int = 1, init_state: List[List[float]] = None, step_size: float = 1.0):
        """Initialisation for the multi-agent system

        Args:
            num_agents (int): Number of agents in the network
            id (int, optional): Identifier of the multi-agent system. Defaults to None.
            graph (nx.Graph, optional): Graph representation of the communication network. Defaults to a random regular graph.
            agents (List[BaseAgent], optional): List of the agents. Defaults to None.
            agent_type (BaseAgent, optional): Type of the agents. Defaults to SingleIntegratorAgent.
            state_dim (int, optional): Dimension of the agents' internal state. Defaults to 1.
            init_state (List[List[float]], optional): Initial states for all agents. Defaults to None.
            step_size (float, optional): Step size for time discretisation. Defaults to 1.0.
        """

        self.num_agents = num_agents
        self.state_dim = state_dim
        self.step_size = step_size
        self.id = id
        if graph is not None:
            self.graph = graph
        else:
            self.graph = nx.random_regular_graph(2, num_agents)
        self.laplacian = nx.laplacian_matrix(graph)
        if init_state is None:
            self.agents = agents if agents is not None else [agent_type(state_dim=state_dim, step_size=step_size) for _ in range(self.num_agents)]
        else:
            self.agents = agents if agents is not None else [agent_type(state_dim=state_dim, step_size=step_size, init_state=init_state[i]) for i in range(self.num_agents)]

        for i in range(self.num_agents):
            for j in range(self.num_agents):
                Lij = self.laplacian[i,j]
                if Lij < 0:
                    self.agents[i].add_neighbours(self.agents[j], -Lij)
        
    def step(self):
        """Proceed one step for all agents
        """
        for agent in self.agents:
            agent.step()

    def reset(self):
        """Reset the multi-agent system
        """
        # TODO: implement the reset function
        pass

    def set_triggering_law(self, trigger: 'EventTrigger'):
        """Set the event-triggering law for all agents

        Args:
            trigger (EventTrigger): Event-triggering law to be set
        """
        for agent in self.agents:
            agent.set_triggering_law(trigger)

    def set_objective_functions(self, objective_functions: List[Callable]):
        """Set the objective functions for each agent

        Args:
            objective_functions (List[Callable]): List of objective functions for each agent, according to the agent id
        """
        self.global_obj = lambda x: np.sum([f(x) for f in objective_functions])
        for i in range(self.num_agents):
            self.agents[i].set_objective_function(objective_functions[i])
    
    def set_dynamics_functions(self, dynamics_functions: List[Callable]):
        """Set the dynamics functions for each agent

        Args:
            dynamics_functions (List[Callable]): [description]
        """
        for i in range(self.num_agents):
            self.agents[i].set_dynamics_function(dynamics_functions[i])

    def get_id(self) -> Union[int, None]:
        """Get the object identifier if it exists

        Returns:
            Union[int, None]: the object identifier
        """
        return self.id

    def get_num_agents(self) -> int:
        """Get the number of agents in the multi-agent system

        Returns:
            int: number of agents in the system
        """
        return self.num_agents

    def get_time(self) -> float:
        """Get the current time of the multi-agent system

        Returns:
            float: current time
        """
        _t = np.array(self.agents[0].stat['t'])
        return _t
    
    def get_step_size(self) -> float:
        """Get the step size of the multi-agent system

        Returns:
            float: step size
        """
        return self.step_size

    def get_states(self) -> List[np.array]:
        """Get the states of all agents

        Returns:
            List[np.array]: List of states of agents
        """
        return [agent.stat['x'] for agent in self.agents]
    
    def get_mean_states(self) -> np.array:
        """Get the mean state of all agents

        Returns:
            np.array: mean state of all agents
        """
        return np.mean([agent.get_state() for agent in self.agents], axis=0)

    def get_current_consensus_error(self) -> float:
        """Get the latest consensus error

        Returns:
            float: latest consensus error
        """
        return self.get_consensus_error()[-1]

    def get_consensus_error(self) -> np.array:
        """Get the consensus error for all time instances

        Returns:
            np.array: consensus error
        """
        _err = 0
        _shape = np.transpose(self.agents[0].stat['x']).shape
        _e = np.zeros(_shape)
        try:
            _e = 0
            for agent in self.agents:
                _e += np.transpose(agent.stat['x'])
            _e = _e / self.num_agents
        except AttributeError:
            _e = 0
            for agent in self.agents:
                _e += np.transpose(agent.stat['x'])
            _e = _e / self.num_agents
        for agent in self.agents:
            _err += (np.transpose(agent.stat['x']) - _e) ** 2
        _err = np.sum(_err, axis=0)
        return _err

    def get_communication_rate(self) -> float:
        """Get the average communication rate for all time instances

        Returns:
            float: average communication rate
        """
        _Gamma = 0
        _gammas = 0
        _t = self.get_time()
        for agent in self.agents:
            _gammas = _gammas + np.array(agent.stat['gamma'])
        _gammas[0] = 0
        _t[0] = 1e-5
        _Gamma = np.cumsum(_gammas) / (self.num_agents * _t)
        _Gamma[0] = 0
        return _Gamma

    def get_state_dim(self) -> int:
        """Get the dimension of agent states

        Returns:
            int: state dimension
        """
        return self.state_dim

    def plot_states(self, index: int = 1, grid: bool = True, xlim: tuple = None, ylim: tuple = None, zoom: bool = True):
        """Plot the agent states

        Args:
            index (int, optional): figure index. Defaults to 1.
            grid (bool, optional): whether of not to display grid. Defaults to True.
            xlim (tuple, optional): display limit on x axis. Defaults to None.
            ylim (tuple, optional): display limit on y axis. Defaults to None.
            zoom (bool, optional): whether of not to create zoomed view. Defaults to True.
        """
        for k in range(self.num_agents):
            agent = self.agents[k]
            _t = agent.stat['t']
            _x = np.transpose(agent.stat['x'])
            for n in range(self.state_dim):
                fig = plt.figure(n+index)
                _xn = _x[n]
                plt.plot(_t, _xn, label='Agent ' + str(k+1))
                plt.xlabel('$t$', fontsize=14)
                plt.ylabel('$x_i^' + str(n+1) + '(t)$', fontsize=14)
        
        try:
            _obj = self.global_obj
            _opt_sol = calculate_convex_optimal(_obj, self.state_dim)
            if self.state_dim == 1:
                _v = np.ones(length(_t)) * _opt_sol
                plt.plot(_t, _v, label='Optimal Value', linestyle='dashdot')
            else:
                for n in range(self.state_dim):
                    plt.figure(n+index)
                    _v = np.ones(length(_t)) * _opt_sol[n]
                    plt.plot(_t, _v, label='Optimal Value', linestyle='dashdot')
                    
        except AttributeError:
            pass

        for n in range(self.state_dim):
            plt.figure(n+index)
            if self.num_agents < 10:
                plt.legend()
            if grid:
                plt.grid(True)
            if xlim != None:
                plt.xlim(xlim)
            if ylim != None:
                plt.ylim(ylim)

        if zoom:
            for n in range(self.state_dim):
                fig = plt.figure(n+index)
                ax_new = fig.add_axes([0.2, 0.175, 0.35, 0.225])
                for k in range(self.num_agents):
                    agent = self.agents[k]
                    _t = agent.stat['t']
                    _x = np.transpose(agent.stat['x'])
                    _xn = _x[n]
                    plt.plot(_t[0:160], _xn[0:160])
                    ax_new.set_xlim([0, 0.4])
                    # ax_new.set_ylim(ylim)

    def plot_consensus_error(self, index = None, loglog = False, semilogy = True, grid = True, label = '', color = None, xlim = None, ylim = None, zoom = True, fig = None, ax1 = None, ax_new = None, zoom_xlim = None, zoom_ylim = None, zoom_pos = [0.325, 0.65, 0.35, 0.225]):
        if index is None:
            index = self.state_dim + 1
        _t = self.get_time()
        _err = self.get_consensus_error()
        # fig = plt.figure(index)
        # ax1 = fig.gca()
        if fig is None and ax1 is None:
            fig = plt.figure(index)
            ax1 = plt.subplot(1,1,1, label='ce')
        # try:
        #     ax1 = fig.axes[0]
        # except IndexError:
        #     ax1 = fig.axes
        
        # plot = plt.semilogy if semilogy else plt.plot
        if loglog:
            plot = ax1.loglog
        elif semilogy:
            plot = ax1.semilogy
        else:
            plot = ax1.plot

        if color != None:
            plot(_t, _err, label=label, color=color)
        else:
            plot(_t, _err, label=label)
        
        ax1.set_xlabel(r'$t$', fontsize=14)
        ax1.set_ylabel(r'$\varepsilon(t)$', fontsize=14)
        if grid:
            ax1.grid(True)
        if label != '' or label != None:
            ax1.legend()

        if xlim != None:
            ax1.set_xlim(xlim)
        if ylim != None:
            ax1.set_ylim(ylim)

        if zoom:
            if ax_new is None:
                ax_new = fig.add_axes(zoom_pos)
            if loglog:
                plot = ax_new.loglog
            elif semilogy:
                plot = ax_new.semilogy
            else:
                plot = ax_new.plot
            # plot = ax_new.plot
            if color != None:
                plot(_t, _err, color=color)
            else:
                plot(_t, _err)

            if zoom_xlim != None:
                ax_new.set_xlim(zoom_xlim)
            else:
                ax_new.set_xlim([0, 2])
            if zoom_ylim != None:
                ax_new.set_ylim(zoom_ylim)
            
        return fig, ax1, ax_new
        

    def plot_communication_rate(self, index = None, grid = True, label = '', color = None, xlim = None, ylim = None, legend_loc = 1, zoom = True, fig = None, ax1 = None, ax_new = None, zoom_xlim = None, zoom_ylim = None, zoom_pos = [0.17, 0.65, 0.35, 0.225]):
        if index is None:
            index = self.state_dim + 2
        _t = self.get_time()
        _Gamma = self.get_communication_rate()

        if fig is None and ax1 is None:
            fig = plt.figure(index)
            ax1 = plt.subplot(1,1,1, label='ce')

        if color != None:
            ax1.plot(_t, _Gamma, label=label, color=color)
        else:
            ax1.plot(_t, _Gamma, label=label)
        ax1.set_xlabel(r'$t$', fontsize=14)
        ax1.set_ylabel(r'$\Gamma(t)$', fontsize=14)
        if grid:
            ax1.grid(True)
        if label != '' or label != None:
            ax1.legend(loc=legend_loc)

        if xlim != None:
            ax1.set_xlim(xlim)
        if ylim != None:
            ax1.set_ylim(ylim)

        if zoom:
            if ax_new is None:
                ax_new = fig.add_axes(zoom_pos)
            if color != None:
                plt.plot(_t, _Gamma, color=color)
            else:
                plt.plot(_t, _Gamma)

            if zoom_xlim != None:
                ax_new.set_xlim(zoom_xlim)
            else:
                ax_new.set_xlim([0, 2])
            if zoom_ylim != None:
                ax_new.set_ylim(zoom_ylim)

        return fig, ax1, ax_new

    def plot_interevent_time(self, index = 100, grid = True, label = '', color = None, xlim = None, ylim = None, legend_loc = 1):
        # TODO: plot the interevent interval
        pass
    
    def draw_graph(self, index: int = None):
        """Display the underlying communication graph of the multi-agent system

        Args:
            index (int, optional): figure index. Defaults to 100.
        """
        plt.figure(index)
        nx.draw(self.graph)
    
    def show_plots(self):
        """Alias function for plt.show() from matplotlib
        """
        plt.show()


