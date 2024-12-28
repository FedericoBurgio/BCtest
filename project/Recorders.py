from typing import Any, Dict, Callable
import numpy as np
import os
import detectContact
import h5py
import copy

class DataRecorderPD(object):
    def __init__(self, controller, record_dir = "") -> None:
        self.record_dir = "datasets/" + record_dir
        self.controller_ = controller
        self.gait_index = -1
 
        # Initialize as empty arrays instead of lists for more efficient storage
        self.s_list = []
        self.qNext_list = []
        #self.cnt_plan = []# testing purposes
       
        print("RECORDER INITIATED")
        
    def record(self, q: np.array, v: np.array, tau: np.array, robot_data: Any, **kwargs) -> None:
        s = np.concatenate((q[2:], v))  #len 35 q = #19-2, v = #18

        # q_copy = q.copy()
        # v_copy = v.copy()
        # q_copy[:2] = 0.

        #t = robot_data.time
        #self.robot.update(q_copy, v_copy)
        #cnt_plan = self.controller_.gait_gen.compute_raibert_contact_plan(q_copy, v_copy, t, self.v_des, self.w_des)
        #self.gait_gen.cnt_plan = cnt_plan
        #self.robot.update(q, v)
        
        cnt_base = detectContact.detect_contact_steps4(self.controller_.gait_gen.cnt_plan.copy(), q) # #17:[4x4],1 _ [[bool1,x1,y1,z1],[bool2,x2,y2,z2],[bool3,x3,y3,z3],[bool4,x4,y4,z4],timesteps]
        #cnt_base = detectContact.detect_contact_steps4(cnt_plan,q) # #17:[4x4],1 _ [[bool1,x1,y1,z1],[bool2,x2,y2,z2],[bool3,x3,y3,z3],[bool4,x4,y4,z4],timesteps]
        
        s = np.append(s, cnt_base[0].flatten()) # actual next contact
        s = np.append(s, cnt_base[1]) # timesteps. Note: NOT expressed in seconds #len 52
        
        #len 52
        tmp = cnt_base[0] #actual bool xyz for ech EE
        tmp[:,1:] = detectContact.express_contact_plan_in_consistant_frame(q, tmp[:,1:],True) #express xyz in base frame
        s = np.append(s, tmp.flatten()) # add bool xyz #16 [4x4] 
        s = np.append(s, 0)#add time step #1
        
        #len 69 
        s = np.append(s, self.controller_.gait_gen.gait_planner.get_percent_in_phase(robot_data.time, 0))
        s = np.append(s, self.controller_.gait_gen.gait_planner.get_percent_in_phase(robot_data.time, 1))
        s = np.append(s, self.controller_.gait_gen.gait_planner.get_percent_in_phase(robot_data.time, 2))
        s = np.append(s, self.controller_.gait_gen.gait_planner.get_percent_in_phase(robot_data.time, 3))
        
        #len 73
        s = np.append(s, self.controller_.gait_gen.gait_planner.get_phase(robot_data.time, 0))
        s = np.append(s, self.controller_.gait_gen.gait_planner.get_phase(robot_data.time, 1))
        s = np.append(s, self.controller_.gait_gen.gait_planner.get_phase(robot_data.time, 2))
        s = np.append(s, self.controller_.gait_gen.gait_planner.get_phase(robot_data.time, 3))
        
        # #len 77
        # s = np.append(s, robot_data.time)
        
        #len 78
        s = np.append(s, self.gait_index)
        s = np.append(s, self.controller_.v_des)# #3
        s = np.append(s, self.controller_.w_des)# #1
        
        #len 82
        
        # s = [q:(0..16), v:(17..34),cntNEXT:(35,51 ##35 bool1 36 x1 37 y1...),cntNOW:(52..68),v_w_des:(67..72)]
        #len s 
        
        tau_ = np.array(list(tau.values()))    # Convert action (tau) to numpy array
        kp = 2
        kv = 0.1
        
        a = np.zeros(12)
        for i in range(12):
            a[i] = q[7+i] + (tau_[i] + kv*v[6+i])/kp# +7: x y z qx qy qz alpha q1 ... q12 q1 index 7; +6 same 
   
        if not np.isnan(a).any() and not np.isinf(a).any():
            self.s_list.append(s)
            self.qNext_list.append(a)

    def save_data_hdf5(self, filename):
        new_states = np.array(self.s_list)
        new_actions = np.array(self.qNext_list)
        with h5py.File(f"datasets/{filename}.h5", 'a') as f:
            if 'states' in f:
                f['states'].resize((f['states'].shape[0] + new_states.shape[0]), axis=0)
                f['states'][-new_states.shape[0]:] = new_states
                f['qNext'].resize((f['qNext'].shape[0] + new_actions.shape[0]), axis=0)
                f['qNext'][-new_actions.shape[0]:] = new_actions
            else:
                f.create_dataset('states', data=new_states, maxshape=(None, new_states.shape[1]), chunks=True)
                f.create_dataset('qNext', data=new_actions, maxshape=(None, new_actions.shape[1]), chunks=True)
        self.s_list.clear()
        self.qNext_list.clear()
  
class DataRecorderNominal(object):
    def __init__(self, controller, record_dir = "") -> None:
        self.record_dir = "datasets/" + record_dir
        self.controller_ = controller
        self.gait_index = -1
 
        # Initialize as empty arrays instead of lists for more efficient storage
        self.s_list = []
        self.v_list = []
        self.cnt_list = []
        
        self.cnt_bools = []
        self.gait_des = []
        #self.cnt_bools2 = []
        print("RECORDER INITIATED")
        
    def record(self, q: np.array, v: np.array, tau: np.array, robot_data: Any, **kwargs) -> None:
        self.s_list.append(q)
        self.v_list.append(v)
        self.cnt_bools.append(self.controller_.gait_gen.cnt_plan[0].flatten()[::4])
        
        tmp = []
        tmp = np.append(tmp, self.gait_index)
        tmp = np.append(tmp, self.controller_.v_des[:2])
        tmp = np.append(tmp, self.controller_.w_des)
        self.gait_des.append(tmp)
        del tmp
        
        tmp = self.controller_.gait_gen.cnt_plan[0].copy()  
        tmp[:,1:] = detectContact.express_contact_plan_in_consistant_frame(q, tmp[:,1:],False) #express xyz in base frame
        self.cnt_list.append(tmp.flatten())
        
        
    def getNominal(self, steps = 1):
        return self.s_list[::steps], self.v_list[::steps], self.cnt_bools[::steps]
    
    def save_data_hdf5(self, filename):
        new_states = np.array(self.s_list)
        new_v = np.array(self.v_list)
        new_cnt_bools = np.array(self.cnt_bools)
        new_gait_des = np.array(self.gait_des)  
        new_cnt = np.array(self.cnt_plan)  
        
        with h5py.File(f"datasets/{filename}.h5", 'a') as f:
            if 'states' in f:
                f['states'].resize((f['states'].shape[0] + new_states.shape[0]), axis=0)
                f['states'][-new_states.shape[0]:] = new_states
                
                f['v'].resize((f['v'].shape[0] + new_v.shape[0]), axis=0)
                f['v'][-new_v.shape[0]:] = new_v
                
                f['cnt_bools'].resize((f['cnt_bools'].shape[0] + new_cnt_bools.shape[0]), axis=0)
                f['cnt_bools'][-new_cnt_bools.shape[0]:] = new_cnt_bools
                
                f['gait_des'].resize((f['gait_des'].shape[0] + new_gait_des.shape[0]), axis=0)
                f['gait_des'][-new_gait_des.shape[0]:] = new_gait_des
                
                f['cnt'].resize((f['cnt'].shape[0] + new_gait_des.shape[0]), axis=0)
                f['cnt'][-new_gait_des.shape[0]:] = new_gait_des
                
            else:
                f.create_dataset('states', data=new_states, maxshape=(None, new_states.shape[1]), chunks=True)
                f.create_dataset('v', data=new_v, maxshape=(None, new_v.shape[1]), chunks=True)
                f.create_dataset('cnt_bools', data=new_cnt_bools, maxshape=(None, new_cnt_bools.shape[1]), chunks=True)
                f.create_dataset('gait_des', data=new_gait_des, maxshape=(None, new_gait_des.shape[1]), chunks=True)
                f.create_dataset('cnt', data=new_cnt, maxshape=(None, new_cnt.shape[1]), chunks=True)
                
        self.s_list.clear()
        self.v_list.clear()
        self.cnt_bools.clear()
        self.gait_des.clear()
        self.cnt_plan.clear()
        
    def save_data_hdf5_tracking_perf(self, filename):
        new_states = np.array(self.s_list)
        new_v = np.array(self.v_list)
        new_cnt = np.array(self.cnt_list)
        #if len(new_states)==100 : breakpoint()
        with h5py.File(f"{filename}", 'w') as f:
            f.create_dataset('states', data=new_states, maxshape=(None, new_states.shape[1]), chunks=True)
            f.create_dataset('v', data=new_v, maxshape=(None, new_v.shape[1]), chunks=True)
            f.create_dataset('cnt', data=new_cnt, maxshape= (None, new_cnt.shape[1]), chunks=True)  
        self.s_list.clear()
        self.v_list.clear()
        self.cnt_list.clear()
        
    # def getNominal2(self, steps):
    #     s_list = []
    #     v_list = []
    #     cnt_bools = []
        
    #     for i in range(len(self.s_list)):
    #         if i % steps == 0:
    #             s_list.append(self.s_list[i])
    #             v_list.append(self.v_list[i])
    #             cnt_bools.append(self.cnt_bools2[i])
    #     return s_list, v_list, cnt_bools