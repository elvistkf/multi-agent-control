import numpy as np
import platform
import matplotlib as mpl
if platform.system() == 'Darwin':       # avoid bugs in some versions of matplotlib with macOS catalina
    mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
from typing import Any, Callable, List
from pathos.multiprocessing import ProcessingPool as Pool

from .multiagentcontrol import MultiagentControl
from .event_trigger import EventTrigger
from .common import length, is_scalar, lowpass_filter

class ParallelEnv():
    """Parallisation of the MultiagentControl class
    """
    def __init__(self, envs: 'List[MultiagentControl]', run: Callable, process: int = 6, final: bool = False):
        """Initialise the parallel environment

        Args:
            envs (List[MultiagentControl]): list of MultiagentControl objects to be parallelised
            run (Callable): function of running the environments in parallel
            process (int, optional): number of parallel processes. Defaults to 6.
            final (bool, optional): [description]. Defaults to False.
        """
        self.__envs = envs
        self.__num_envs = length(envs)
        self.__run = run
        self.__process = process
        self.__state_dim = self.__envs[0].get_state_dim()
        self.__final = final
        self.__step_size = self.__envs[0].get_step_size()
    
    def run(self):
        """Running the processes in parallel
        """
        with Pool(process=self.__process) as pool:
            self.result = pool.map(self.__run, self.__envs)
        self.__envs = self.result
        
    def set_triggering_laws(self, triggers: 'List[EventTrigger]'):
        """Set the event-triggering laws for each environment

        Args:
            triggers (List[EventTrigger]): list of event-triggers

        Raises:
            ValueError: The list of trigger laws does not match the number of environments
        """
        if is_scalar(triggers):
            for env in self.__envs:
                env.set_triggering_law(triggers)
        elif length(triggers) == length(self.__envs):
            for i in range(length(self.__envs)):
                env = self.__envs[i]
                trigger = triggers[i]
                env.set_triggering_law(trigger)
        else:
            raise ValueError('The list of trigger laws does not match the number of environments')

    def set_objective_functions(self, objective_functions: List[Callable]):
        """Set the objective functions for each environment, if applicable

        Args:
            objective_functions (List[Callable]): list of objective functions to be assigned
        """
        for env in self.__envs:
            env.set_objective_functions(objective_functions)

    def get_envs(self) -> List[MultiagentControl]:
        """Get the environments

        Returns:
            List[MultiagentControl]: list of MultiagentControl objects
        """
        return self.__envs

    def get_consensus_errors(self) -> List[float]:
        """Get the consensus errors for each environment

        Returns:
            List[float]: list of consensus errors
        """
        return [env.get_consensus_error() for env in self.__envs]

    def plot_states(self, index = 1, grid = True):
        """Plot the states of the environment of specific index

        Args:
            index (int, optional): index of target environment. Defaults to 1.
            grid (bool, optional): whether or not to display grid. Defaults to True.
        """
        self.__envs[0].plot_states(index, grid)

    def plot_consensus_error(self, index: int = None, grid: bool = True, loglog: bool = False, semilogy: bool = True, fill: bool = True, label: str = '', color:str = None, xlim: tuple = None, ylim: tuple = None, zoom: bool = True, fig = None, ax1 = None, ax_new = None, zoom_xlim: tuple = None, zoom_ylim: tuple = None, zoom_pos: List[float] = [0.215, 0.65, 0.35, 0.225]) -> Any:
        """Plot the consensus error of the multi-agent control systems

        Args:
            index (int, optional): figure index. Defaults to None.
            grid (bool, optional): display grid. Defaults to True.
            loglog (bool, optional): loglog scale. Defaults to False.
            semilogy (bool, optional): semilog-y scale. Defaults to True.
            fill (bool, optional): fill the area between max-min range. Defaults to True.
            label (str, optional): figure label. Defaults to ''.
            color (str, optional): plot colour. Defaults to None.
            xlim (tuple, optional): range limit on x axis. Defaults to None.
            ylim (tuple, optional): range limit on y axis. Defaults to None.
            zoom (bool, optional): display zoomed view. Defaults to True.
            fig ([type], optional): figure to superpose on. Defaults to None.
            ax1 ([type], optional): axis to superpose on. Defaults to None.
            ax_new ([type], optional): new axis. Defaults to None.
            zoom_xlim (tuple, optional): range limit on x axis of the zoomed view. Defaults to None.
            zoom_ylim (tuple, optional): range limit on y axis of the zoomed view. Defaults to None.
            zoom_pos (List[float], optional): zoomed view position. Defaults to [0.215, 0.65, 0.35, 0.225].

        Returns:
            Any: figure, ax1, ax_new
        """
        if index is None:
            index = self.__state_dim + 1
        _t = self.__envs[0].get_time()
        _errs = np.array([env.get_consensus_error() for env in self.__envs])
        return self.plot_data_mean(_t, _errs, index, grid, loglog = loglog, semilogy = semilogy, fill=fill, ylabel=r'$\varepsilon(t)$', label=label, color=color, zoom=zoom, fig=fig, ax1=ax1, ax_new=ax_new, zoom_xlim=zoom_xlim, zoom_ylim=zoom_ylim, zoom_pos=zoom_pos)

    def plot_communication_rate(self, index: int = None, grid: bool = True, fill: bool = True, label: str = '', color: str = None, xlim: tuple = None, ylim: tuple = None, zoom: bool = True, fig = None, ax1 = None, ax_new = None, zoom_xlim: tuple = None, zoom_ylim: tuple = None, zoom_pos: List[float] = [0.19, 0.495, 0.35, 0.35]):
        if index is None:
            index = self.__state_dim + 2
        _t = self.__envs[0].get_time()
        _Gammas = np.array([env.get_communication_rate() for env in self.__envs])
        # self.plot_data_mean(_t, _Gammas, index, grid, ylabel=r'$\Gamma(t)$', std_filter = True)
        return self.plot_data_mean(_t, _Gammas, index, grid, semilogy=False, fill=fill, ylabel=r'$\Gamma(t)$', std_filter = False, label=label, color=color, zoom=zoom, fig=fig, ax1=ax1, ax_new=ax_new, zoom_xlim=zoom_xlim, zoom_ylim=zoom_ylim, zoom_pos=zoom_pos)

    def draw_graph(self, index: int = 100):
        """Display the communication graph

        Args:
            index (int, optional): figure index. Defaults to 100.
        """
        self.__envs[0].draw_graph(index)

    def plot_data_mean(self, x: np.array, y: np.array, index: int = 1, grid: bool = True, loglog: bool = False, semilogy: bool = False, fill: bool = True, xlabel: str = r'$t$', ylabel: str = '', label: str = '', title: str = '', color: str = None, allow_neg: bool = False, std_filter: bool = False, xlim: bool = None, ylim: bool = None, legend_loc: int = 1, zoom: bool = True, fig = None, ax1 = None, ax_new = None, zoom_xlim: tuple = None, zoom_ylim: tuple = None, zoom_pos: List[float] = None):

        _t = x
        _data = y
        _data_mean = np.mean(_data, axis=0)
        _data_std = np.std(_data, axis=0)
        _data_max = np.max(_data, axis=0)
        _data_min = np.min(_data, axis=0)
        
        if fig is None and ax1 is None:
            fig = plt.figure(index)
            ax1 = plt.subplot(1,1,1, label='ce')
        
        if loglog:
            plot = ax1.loglog
        elif semilogy:
            plot = ax1.semilogy
        else:
            plot = ax1.plot
        if color != None:
            plot(_t, _data_mean, label=label, color=color)
        else:
            plot(_t, _data_mean, label=label)
        
        if fill:
            if std_filter:
                _data_std = lowpass_filter(_data_std)
            low = _data_mean - 1 * _data_std
            low = low if allow_neg else np.clip(low, 0, None)
            low += _data_min
            low /= 2
            high = _data_mean + 1 * _data_std
            high += _data_max
            high/= 2
            if std_filter:
                low = lowpass_filter(low)
                high = lowpass_filter(high)
            if color != None:
                ax1.fill_between(_t, low, high, alpha=0.4, color=color)
            else:
                ax1.fill_between(_t, low, high, alpha=0.4)

        ax1.set_xlabel(xlabel, fontsize=14)
        ax1.set_ylabel(ylabel, fontsize=14)
        if grid:
            ax1.grid(True)
        if label != '' or label != None:
            ax1.legend(loc=legend_loc)

        if xlim != None:
            plt.xlim(xlim)
        if ylim != None:
            plt.ylim(ylim)

        if zoom:
            if ax_new is None:
                ax_new = fig.add_axes(zoom_pos)
            if loglog:
                plot = ax_new.loglog
            elif semilogy:
                plot = ax_new.semilogy
            else:
                plot = ax_new.plot
            if color != None:
                plot(_t, _data_mean, label=label, color=color)
            else:
                plot(_t, _data_mean, label=label)
            if fill:
                if std_filter:
                    _data_std = lowpass_filter(_data_std)
                low = _data_mean - 1 * _data_std
                low = low if allow_neg else np.clip(low, 0, None)
                low += _data_min
                low /= 2
                low = _data_min

                high = _data_mean + 1 * _data_std
                high += _data_max
                high/= 2
                high = _data_max
                if color != None:
                    ax_new.fill_between(_t, low, high, alpha=0.4, color=color)
                else:
                    ax_new.fill_between(_t, low, high, alpha=0.4)

            if zoom_xlim != None:
                ax_new.set_xlim(zoom_xlim)
            else:
                ax_new.set_xlim([0, 2])
            if zoom_ylim != None:
                ax_new.set_ylim(zoom_ylim)
        
        return fig, ax1, ax_new

    def show_plots(self):
        """Alias function for plot.show() from matplotlib
        """
        plt.show()