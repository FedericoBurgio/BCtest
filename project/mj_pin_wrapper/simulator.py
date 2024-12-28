# TUM - MIRMI - ATARI lab
# Victor DHEDIN, 2024

from typing import Any, Callable
import mujoco
from mujoco import viewer
import time
import numpy as np

from mj_pin_wrapper.mj_robot import MJQuadRobotWrapper
#from mj_pin_wrapper.mj_pin_robot import MJPinQuadRobotWrapper
from mj_pin_wrapper.abstract.controller import ControllerAbstract
from mj_pin_wrapper.abstract.data_recorder import DataRecorderAbstract
import pinocchio as pin
from pinocchio.utils import zero

import numpy as np
class Simulator(object):
    DEFAULT_SIM_DT = 1.0e-3 #s
    def __init__(self,
                 robot,#: MJPinQuadRobotWrapper,
                 controller: ControllerAbstract = None,
                 data_recorder: DataRecorderAbstract = None,
                 sim_dt : float = 1.0e-3,
                 ) -> None:
        
        self.robot = robot.mj
        self.pin_wrapper = robot
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
        mode = kwargs.get("mode", -1)
        comb = kwargs.get("comb", [])
        fails = kwargs.get("fails", 0)
        pertStep = kwargs.get("pertStep", -1)
        pertNomqs = kwargs.get("pertNomqs", [])
        pertNomvs = kwargs.get("pertNomvs", [])
        
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
        self.verbose = kwargs.get("verbose", False)
        self.stop_on_collision = kwargs.get("stop_on_collision", False)
        self.visual_callback_fn = visual_callback_fn
        
        def apply_perturbation():
            # Extract the model and data from the wrapper
            q = self.robot.get_state()[0]   
            v = self.robot.get_state()[1]
            cntBools= self.controller.gait_gen.cnt_plan[0].flatten()[::4]
            #self.pin_wrapper = self.robot
            pin_model = self.pin_wrapper.pin.model
            data = self.pin_wrapper.pin.data

            self.pin_wrapper.reset(q, v)
            pin.computeJointJacobians(pin_model, data, q)
            #pin.forwardKinematics(pin_model, data, q, v)

            nq = pin_model.nq  # Number of configuration variables (19)
            nv = pin_model.nv  # Number of velocity variables (18)

            # Frame IDs of the end-effectors (modify these IDs according to your robot's end-effectors)
            EE_frames_all = np.array([14, 26, 42, 54], dtype=int)  # Ensure dtype is int
            EE_frames = EE_frames_all[cntBools == 1]

            cnt_jac = np.zeros((3*len(EE_frames), nv))
            cnt_jac_dot = np.zeros((3*len(EE_frames), nv))

            def rotate_jacobian(jac, index):
                world_R_joint = pin.SE3(data.oMf[index].rotation, zero(3))
                return world_R_joint.action @ jac

            for ee_cnt in range(len(EE_frames)):

                jac = pin.getFrameJacobian(pin_model,\
                    data,\
                    int(EE_frames[ee_cnt]),\
                    pin.ReferenceFrame.LOCAL)
                cnt_jac[3*ee_cnt:3*(ee_cnt+1),] = rotate_jacobian(jac,int(EE_frames[ee_cnt]))[0:3,]
                jac_dot = pin.getFrameJacobianTimeVariation(pin_model,\
                    data,\
                    int(EE_frames[ee_cnt]),\
                    pin.ReferenceFrame.LOCAL)
                cnt_jac_dot[3*ee_cnt:3*(ee_cnt+1),] = rotate_jacobian(jac_dot,int(EE_frames[ee_cnt]))[0:3,]

                min_ee_height=.0
            while min_ee_height >= 0:
                #np.random.seed(seed=seed)
                perturbation_pos = np.concatenate((
                    np.random.uniform(-0.15, 0.15, 3),
                    np.random.uniform(-0.15, 0.15, 3),
                    np.random.uniform(-0.32, 0.32, nv - 6)
                ))

                perturbation_vel = np.random.uniform(-0.18, 0.18, nv)
                if EE_frames.size == 0:
                    random_pos_vec = perturbation_pos
                    random_vel_vec = perturbation_vel
                else:
                    random_pos_vec = (np.identity(nv) - np.linalg.pinv(cnt_jac)@\
                                cnt_jac) @ perturbation_pos
                    jac_vel = cnt_jac_dot * perturbation_pos + cnt_jac * perturbation_vel
                    random_vel_vec = (np.identity(nv) - np.linalg.pinv(jac_vel)@\
                                jac_vel) @ perturbation_pos

                ### add perturbation to nominal trajectory
                v0_ = v + random_vel_vec
                q0_ = pin.integrate(pin_model, \
                    q, random_pos_vec)

                ### check if the swing foot is below the ground
                # pin.forwardKinematics(pin_model, data, q0_, v0_)
                # self.pin_wrapper.reset(q0_, v0_)
                # pin.framesForwardKinematics(pin_model, data, q)
                # pin.updateFramePlacements(pin_model, data)
                ee_below_ground = []
                self.pin_wrapper.reset(q0_, v0_)
                for e in range(len(EE_frames_all)):
                    frame_id = int(EE_frames_all[e])
                    if data.oMf[frame_id].translation[2] < 0.0001:
                        ee_below_ground.append(1)
                if len(ee_below_ground) == 0: # ee_below_ground==[]
                    min_ee_height = -1.

            return q0_, v0_
            
        if len(comb) != 0:
            from mpc_controller.motions.cyclic.go2_jump import jump
            from mpc_controller.motions.cyclic.go2_trot import trot
            from mpc_controller.motions.cyclic.go2_bound import bound
            gaits = [trot, jump, bound]
            switch_step = simulation_time * 1000
            simulation_time *= len(comb)
        
        if self.verbose:
            print("-----> Simulation start")
        
        self.sim_step = 0
        self.robot.xfrc_applied = []
        self.gait_index = -1
        
        def resetToPerturbation():
            if pertStep > 0:
                if self.sim_step % pertStep == 0: #and self.sim_step > 0:
                    #breakpoint()
                    self.robot.update(pertNomqs[self.sim_step//pertStep], pertNomvs[self.sim_step//pertStep])
                                        
                    print("Perturbation applied")
        
        def randomForce(selfRobotData, timing, f):
            if randomize: #nota per recoording: usare self.data_recorder.gait_index, per testing trained usare self.controller.gait_index 
                if self.gait_index == 1: f=f*0.5
                if self.gait_index == 2: f=f*0.8
                for i in range(len(timing)):
                    if self.sim_step % timing[i] == 0:
                        body_index = np.random.randint(14)
                        if body_index == 1: #nota per recoording: usare self.data_recorder.gait_index, per testing trained usare self.controller.gait_index 
                            selfRobotData.xfrc_applied[body_index] = 2.5*np.random.uniform(-f[i], f[i], 6)
                        else:
                            selfRobotData.xfrc_applied[body_index] = np.random.uniform(-f[i], f[i], 6)

        def RTswitch():
            if len(comb) != 0 and self.sim_step % switch_step == 0:
                iteration = int(self.sim_step / switch_step)
                v_des = np.zeros(3) 
                #breakpoint()
                self.gait_index = comb[iteration][0]
                sel_gait = gaits[int(self.gait_index)]    
                v_des[0:2] = comb[iteration][1:3] 
                w_des = comb[iteration][3]#np.random.uniform(-0.02, 0.02)
                self.controller.set_command(v_des, w_des)
                self.controller.set_gait_params(sel_gait)
                print("Switching, step: ",self.sim_step )
                    
        self.timing = np.array([np.random.randint(15,25), np.random.randint(180,220), np.random.randint(550,650)])
        #self.timing = np.array([np.random.randint(23,33)])#, np.random.randint(850,950)])
        self.f=np.array([25, 300, 850])   
        #self.f=np.array((18])#, 350])   
        # With viewer
        if use_viewer:
            #for i in range(14):
                #print('name of geom ', i, ': ', self.robot.model.body(i).name) 
            
            with mujoco.viewer.launch_passive(self.robot.model, self.robot.data) as viewer:
                            
                # Enable wireframe rendering of the entire scene.
                viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
                viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0
                viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
                
                viewer.sync()
                sim_start_time = time.time()
                
                while (viewer.is_running() and #reminder: la differenza in applicare la forza qua o dove non si usa il viewer è che self.robot.data.xfrc_applied qua resta con la forza (gneeralizzata) applicata un unico time step, mentre se la applico doe non c'è il viewer resta applicata ( per sempre(?))
                    (simulation_time < 0. or
                        self.sim_step * self.sim_dt < simulation_time)):
                    RTswitch()
                    randomForce(self.robot.data, self.timing, (self.f)*(1-fails/5))
                    #print("Step: ", self.sim_step)
                    #if self.sim_step == 100: breakpoint()
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
               
                RTswitch()
                randomForce(self.robot.data, self.timing, (self.f)*(1-fails/5))
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