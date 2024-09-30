# TUM - MIRMI - ATARI lab
# Victor DHEDIN, 2024

from typing import Any, Callable
import mujoco
from mujoco import viewer
import time
import numpy as np

from mj_pin_wrapper.mj_robot import MJQuadRobotWrapper
from mj_pin_wrapper.abstract.controller import ControllerAbstract
from mj_pin_wrapper.abstract.data_recorder import DataRecorderAbstract


import numpy as np
class Simulator(object):
    DEFAULT_SIM_DT = 1.0e-3 #s
    def __init__(self,
                 robot: MJQuadRobotWrapper,
                 controller: ControllerAbstract = None,
                 data_recorder: DataRecorderAbstract = None,
                 sim_dt : float = 1.0e-3,
                 ) -> None:
        
        self.robot = robot
        self.controller = (controller
                            if controller != None
                            else ControllerAbstract(robot)
                            )
        #self.controller = ControllerAbstract(robot)
        self.data_recorder = (data_recorder
                              if data_recorder != None
                              else DataRecorderAbstract()
                              )
        #self.data_recorder=DataRecorderAbstract()
        self.sim_dt = sim_dt
        self.robot.model.opt.timestep = sim_dt

        # Timings
        self.sim_step = 0
        self.simulation_it_time = []
        self.visual_callback_fn = None
        self.verbose = False
        self.stop_sim = False
        
    def _reset(self) -> None:
        """
        Reset flags and timings.
        """
        self.sim_step = 0
        self.simulation_it_time = []
        self.verbose = False
        self.stop_sim = False
        
    def _simulation_step(self) -> None:
        """
        Main simulation step.
        - Record data
        - Compute and apply torques
        - Step simulation
        """
        # Get state in Pinocchio format (x, y, z, qx, qy, qz, qw)
        self.q, self.v = self.robot.get_state()

        
        
        # Torques should be a map {joint_name : torque value}
        torques = self.controller.get_torques(self.q,
                                              self.v,
                                              robot_data = self.robot.data)
        
        # Record data
        self.data_recorder.record(self.q,
                                  self.v,
                                  torques,
                                  self.robot.data)
        
        # Apply torques
        self.robot.send_joint_torques(torques)
        # Sim step
        mujoco.mj_step(self.robot.model, self.robot.data)
        self.robot.contact_updated = False
        self.sim_step += 1
        
        # TODO: Add external disturbances
        
    def _simulation_step_with_timings(self,
                                      real_time: bool,
                                      ) -> None:
        """
        Simulation step with time keeping and timings measurements.
        """
        
        step_start = time.time()
        self._simulation_step()
        step_duration = time.time() - step_start
        
        self.simulation_it_time.append(step_duration)

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = self.sim_dt - step_duration
        if real_time and time_until_next_step > 0:
            time.sleep(time_until_next_step)
            
    def _stop_sim(self) -> bool:
        """
        True if the simulation has to be stopped.

        Returns:
            bool: stop simulation
        """        
        if self.stop_on_collision and (self.robot.collided or self.robot.is_collision()):
            if self.verbose: print("/!\ Robot collision")
            return True

        if self.stop_sim:
            if self.verbose: print("/!\ Simulation stopped")
            return True
        
        if self.controller.diverged:
            if self.verbose: print("/!\ Controller diverged")
            return True
        
        return False
        
    def run(self,
            simulation_time: float = -1.,
            use_viewer: bool = True,
            visual_callback_fn: Callable = None,
            **kwargs,
            ) -> None:
        randomize = kwargs.get("randomize", False)
        """
        Run simulation for <simulation_time> seconds with or without a viewer.

        Args:
            - simulation_time (float, optional): Simulation time in second.
            Unlimited if -1. Defaults to -1.
            - visual_callback_fn (fn): function that takes as input:
                - the viewer
                - the simulation step
                - the state
                - the simulation data
            that create visual geometries using the mjv_initGeom function.
            See https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer
            for an example.
            - viewer (bool, optional): Use viewer. Defaults to True.
            - verbose (bool, optional): Print timing informations.
            - stop_on_collision (bool, optional): Stop the simulation when there is a collision.
        """
        real_time = kwargs.get("real_time", use_viewer)
        self.verbose = kwargs.get("verbose", True)
        self.stop_on_collision = kwargs.get("stop_on_collision", False)
        self.visual_callback_fn = visual_callback_fn
        
        if self.verbose:
            print("-----> Simulation start")
        
        self.sim_step = 0
        self.robot.xfrc_applied = []
        # With viewer
        def randomForce(selfRobotData):
            timing = [30, 120, 970]
            f = [15, 255, 990]
            if randomize:
                for i in range(len(timing)):
                    if self.sim_step % timing[i] == 0:
                        selfRobotData.xfrc_applied[np.random.randint(14)] = np.random.uniform(-f[i], f[i], 6)

        
        if use_viewer:
            for i in range(14):
                print('name of geom ', i, ': ', self.robot.model.body(i).name)    
            with mujoco.viewer.launch_passive(self.robot.model, self.robot.data) as viewer:
                            
                            # Enable wireframe rendering of the entire scene.
                viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
                viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0
                viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
                
                viewer.sync()
                sim_start_time = time.time()
                
                applied = False
                while (viewer.is_running() and #reminder: la differenza in applicare la forza qua o dove non si usa il viewer è che self.robot.data.xfrc_applied qua resta con la forza (gneeralizzata) applicata un unico time step, mentre se la applico doe non c'è il viewer resta applicata ( per sempre(?))
                    (simulation_time < 0. or
                        self.sim_step * self.sim_dt < simulation_time)):
                    
                    randomForce(self.robot.data)
                    self._simulation_step_with_timings(real_time)
                    self.update_visuals(viewer)
                    viewer.sync()

                
                    if self._stop_sim():
                        break
                    
        # No viewer
        else:
            sim_start_time = time.time()
            while (simulation_time < 0. or self.sim_step < simulation_time * (1 / self.sim_dt)):
                if np.any(self.robot.data.xfrc_applied != 0):
                    self.robot.data.xfrc_applied = np.zeros_like(self.robot.data.xfrc_applied)   
                
                randomForce(self.robot.data)
                self._simulation_step_with_timings(real_time)
                
                
                if self._stop_sim():
                    break
    
        if self.verbose:
            print(f"-----> Simulation end\n")
            sum_step_time = sum(self.simulation_it_time)
            mean_step_time = sum_step_time / len(self.simulation_it_time)
            total_sim_time = time.time() - sim_start_time
            print(f"--- Total optimization step time: {sum_step_time:.2f} s")
            print(f"--- Mean simulation step time: {mean_step_time*1000:.2f} ms")
            print(f"--- Total simulation time: {total_sim_time:.2f} s")
        
        
        # Reset flags
        self._reset()
        
        # TODO: Record video
        
    def update_visuals(self, viewer) -> None:
        """
        Update visuals according to visual_callback_fn.
        
        Args:
            viewer (fn): Running MuJoCo viewer.
        """
        if self.visual_callback_fn != None:
            try:
                self.visual_callback_fn(viewer, self.sim_step, self.q, self.v, self.robot.data)
                
            except Exception as e:
                if self.verbose:
                    print("Can't update visual geometries.")
                    print(e)