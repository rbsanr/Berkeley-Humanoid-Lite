import time
import numpy as np
import torch
import argparse
from loop_rate_limiters import RateLimiter
import recoil as recoil
import threading
from berkeley_humanoid_lite_lowlevel.policy.config import Cfg
import serial

class wholebody_operator:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.rate = RateLimiter(frequency=200.0)

        self.imu = serial.Serial(
            port='/dev/ttyIMU',
            baudrate=115200,
            timeout=1
        )

        self.arm_right   = recoil.Bus('can0', bitrate=500000)
        self.arm_left    = recoil.Bus('can1', bitrate=500000)
        self.leg_right   = recoil.Bus('can2', bitrate=500000)
        self.leg_left    = recoil.Bus('can3', bitrate=500000)

        self.HUMANOID_ARM_LEFT_LITE_JOINTS = [
            "arm_left_shoulder_pitch_joint",
            "arm_left_shoulder_roll_joint",
            "arm_left_shoulder_yaw_joint",
            "arm_left_elbow_pitch_joint",
            "arm_left_elbow_roll_joint",
        ]

        self.HUMANOID_ARM_RIGHT_LITE_JOINTS = [
            "arm_right_shoulder_pitch_joint",
            "arm_right_shoulder_roll_joint",
            "arm_right_shoulder_yaw_joint",
            "arm_right_elbow_pitch_joint",
            "arm_right_elbow_roll_joint",
        ]

        self.HUMANOID_LEG_LEFT_LITE_JOINTS = [
            "leg_left_hip_roll_joint",
            "leg_left_hip_yaw_joint",
            "leg_left_hip_pitch_joint",
            "leg_left_knee_pitch_joint",
            "leg_left_ankle_pitch_joint",
            "leg_left_ankle_roll_joint",
        ]

        self.HUMANOID_LEG_RIGHT_LITE_JOINTS = [
            "leg_right_hip_roll_joint",
            "leg_right_hip_yaw_joint",
            "leg_right_hip_pitch_joint",
            "leg_right_knee_pitch_joint",
            "leg_right_ankle_pitch_joint",
            "leg_right_ankle_roll_joint",
        ]

        self.HUMANOID_LITE_JOINTS = (
            self.HUMANOID_ARM_LEFT_LITE_JOINTS +
            self.HUMANOID_ARM_RIGHT_LITE_JOINTS +
            self.HUMANOID_LEG_LEFT_LITE_JOINTS +
            self.HUMANOID_LEG_RIGHT_LITE_JOINTS
        )

        self.HUMANOID_LITE_ARM_LEFT_JOINTS_INFO = {
            "arm_left_shoulder_pitch_joint" :[self.arm_left,    1,  2.32,  2.32,    1],
            "arm_left_shoulder_roll_joint"  :[self.arm_left,    2,  3.45,  15.59,   1],
            "arm_left_shoulder_yaw_joint"   :[self.arm_left,    3,  3.00,  3.00,    1],
            "arm_left_elbow_pitch_joint"    :[self.arm_left,    4,  1.89,  1.89,    1],
            "arm_left_elbow_roll_joint"     :[self.arm_left,    5,  4.51,  4.51,    1],
        }

        self.HUMANOID_LITE_ARM_RIGHT_JOINTS_INFO = {
            "arm_right_shoulder_pitch_joint":[self.arm_right,   1,  4.73,  4.73,    -1],
            "arm_right_shoulder_roll_joint" :[self.arm_right,   2,  2.56,  15.70,   -1],
            "arm_right_shoulder_yaw_joint"  :[self.arm_right,   3,  2.20,  2.80,    -1],
            "arm_right_elbow_pitch_joint"   :[self.arm_right,   4,  4.64,  4.64,    -1],
            "arm_right_elbow_roll_joint"    :[self.arm_right,   5,  1.20,  1.20,    -1],
        }

        self.HUMANOID_LITE_LEG_LEFT_JOINTS_INFO = {
            "leg_left_hip_roll_joint"       :[self.leg_left,    6,  3.33,  3.33,    1],
            "leg_left_hip_yaw_joint"        :[self.leg_left,    1,  5.83,  5.83,    -1],
            "leg_left_hip_pitch_joint"      :[self.leg_left,    2,  1.81,  1.81,    1],
            "leg_left_knee_pitch_joint"     :[self.leg_left,    3,  1.68,  1.68,    1],
            "leg_left_ankle_pitch_joint"    :[self.leg_left,    4,  3.14,  -3.14,   1],
            "leg_left_ankle_roll_joint"     :[self.leg_left,    5,  1.27,  4.41,    1],
        }

        self.HUMANOID_LITE_LEG_RIGHT_JOINTS_INFO = {
            "leg_right_hip_roll_joint"      :[self.leg_right,   6,  3.18,  3.18,    -1],
            "leg_right_hip_yaw_joint"       :[self.leg_right,   1,  3.14,  3.14,    1],
            "leg_right_hip_pitch_joint"     :[self.leg_right,   2,  2.47,  2.47,    1],
            "leg_right_knee_pitch_joint"    :[self.leg_right,   3,  1.36,  1.36,    1],
            "leg_right_ankle_pitch_joint"   :[self.leg_right,   4,  3.28,  6.42,    1],
            "leg_right_ankle_roll_joint"    :[self.leg_right,   5,  3.82,  6.96,    1],
        }

        self.HUMANOID_LITE_JOINTS_INFO = (
            self.HUMANOID_LITE_ARM_LEFT_JOINTS_INFO |
            self.HUMANOID_LITE_ARM_RIGHT_JOINTS_INFO |
            self.HUMANOID_LITE_LEG_LEFT_JOINTS_INFO |
            self.HUMANOID_LITE_LEG_RIGHT_JOINTS_INFO
        )

        self.HUMANOID_LITE_ARM_LEFT_JOINTS_CURRPOS = {
            "arm_left_shoulder_pitch_joint" :0.0,
            "arm_left_shoulder_roll_joint"  :0.0,
            "arm_left_shoulder_yaw_joint"   :0.0,
            "arm_left_elbow_pitch_joint"    :0.0,
            "arm_left_elbow_roll_joint"     :0.0,
        }

        self.HUMANOID_LITE_ARM_RIGHT_JOINTS_CURRPOS = {
            "arm_right_shoulder_pitch_joint":0.0,
            "arm_right_shoulder_roll_joint" :0.0,
            "arm_right_shoulder_yaw_joint"  :0.0,
            "arm_right_elbow_pitch_joint"   :0.0,
            "arm_right_elbow_roll_joint"    :0.0,
        }

        self.HUMANOID_LITE_LEG_LEFT_JOINTS_CURRPOS = {
            "leg_left_hip_roll_joint"       :0.0,
            "leg_left_hip_yaw_joint"        :0.0,
            "leg_left_hip_pitch_joint"      :0.0,
            "leg_left_knee_pitch_joint"     :0.0,
            "leg_left_ankle_pitch_joint"    :0.0,
            "leg_left_ankle_roll_joint"     :0.0,
        }

        self.HUMANOID_LITE_LEG_RIGHT_JOINTS_CURRPOS = {
            "leg_right_hip_roll_joint"      :0.0,
            "leg_right_hip_yaw_joint"       :0.0,
            "leg_right_hip_pitch_joint"     :0.0,
            "leg_right_knee_pitch_joint"    :0.0,
            "leg_right_ankle_pitch_joint"   :0.0,
            "leg_right_ankle_roll_joint"    :0.0,
        }

        self.HUMANOID_LITE_JOINTS_CURRPOS = (
            self.HUMANOID_LITE_ARM_LEFT_JOINTS_CURRPOS |
            self.HUMANOID_LITE_ARM_RIGHT_JOINTS_CURRPOS |
            self.HUMANOID_LITE_LEG_LEFT_JOINTS_CURRPOS |
            self.HUMANOID_LITE_LEG_RIGHT_JOINTS_CURRPOS
        )

        self.HUMANOID_LITE_ARM_LEFT_JOINTS_GOALPOS = {
            "arm_left_shoulder_pitch_joint" :0.0,
            "arm_left_shoulder_roll_joint"  :0.0,
            "arm_left_shoulder_yaw_joint"   :0.0,
            "arm_left_elbow_pitch_joint"    :0.0,
            "arm_left_elbow_roll_joint"     :0.0,
        }

        self.HUMANOID_LITE_ARM_RIGHT_JOINTS_GOALPOS = {
            "arm_right_shoulder_pitch_joint":0.0,
            "arm_right_shoulder_roll_joint" :0.0,
            "arm_right_shoulder_yaw_joint"  :0.0,
            "arm_right_elbow_pitch_joint"   :0.0,
            "arm_right_elbow_roll_joint"    :0.0,
        }

        self.HUMANOID_LITE_LEG_LEFT_JOINTS_GOALPOS = {
            "leg_left_hip_roll_joint"       :0.0,
            "leg_left_hip_yaw_joint"        :0.0,
            "leg_left_hip_pitch_joint"      :0.0,
            "leg_left_knee_pitch_joint"     :0.0,
            "leg_left_ankle_pitch_joint"    :0.0,
            "leg_left_ankle_roll_joint"     :0.0,
        }

        self.HUMANOID_LITE_LEG_RIGHT_JOINTS_GOALPOS = {
            "leg_right_hip_roll_joint"      :0.0,
            "leg_right_hip_yaw_joint"       :0.0,
            "leg_right_hip_pitch_joint"     :0.0,
            "leg_right_knee_pitch_joint"    :0.0,
            "leg_right_ankle_pitch_joint"   :0.0,
            "leg_right_ankle_roll_joint"    :0.0,
        }

        self.HUMANOID_LITE_JOINTS_GOALPOS = (
            self.HUMANOID_LITE_ARM_LEFT_JOINTS_GOALPOS |
            self.HUMANOID_LITE_ARM_RIGHT_JOINTS_GOALPOS |
            self.HUMANOID_LITE_LEG_LEFT_JOINTS_GOALPOS |
            self.HUMANOID_LITE_LEG_RIGHT_JOINTS_GOALPOS
        )

        self.motor_param = {
            "kp":0.3,
            "kd":0.01,
            "ki":0.0,
            "torque":0.2,
            "velocity":0.5,
            "velocity_slow":0.01
        }

        self.mode = 3.0
        self.command_velocity_x = 0.0
        self.command_velocity_y = 0.0
        self.command_velocity_yaw = 0.0

        self.thread_run = True
        motor_feeder = threading.Thread(target=self._feed_motor)
        # motor_feeder.start()

        self.observation = {
            "POSITION":[0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0],
            "VELOCITY":[0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0]
        }

        self.joint_tracking = True
        self.joint_controler = threading.Thread(target=self._joint_pulling)
        self.set_IDLE()
        
        

    def reset(self):
        for index, name in enumerate(self.HUMANOID_LITE_JOINTS):
            pos_obs, vel_obs = self._move_motor(name, self.cfg.default_joint_positions[index])
        msg_reset = '<reset>'
        data_to_send = msg_reset.encode('utf-8')
        bytes_written = self.imu.write(data_to_send)
        if bytes_written == False:
            print("Sending Faile")

    def step(self, action):
        pos_list = []
        vel_list = []

        for index, name in enumerate(self.HUMANOID_LITE_JOINTS):
            origin, dir = self.HUMANOID_LITE_JOINTS_INFO[name][3:]
            # pos_obs, vel_obs = self._move_motor(name, (action[index]*dir) + origin)
            # self.HUMANOID_LITE_JOINTS_GOALPOS[name] = (action[index]*dir)+origin
            print((action[index]*dir)+origin)

        quat, ang_vel, _ = self._get_imu()
        
        # return torch.cat([
        #     torch.tensor(quat),
        #     torch.tensor(ang_vel),
        #     # torch.tensor(pos_list)[self.cfg.action_indices],
        #     # torch.tensor(vel_list)[self.cfg.action_indices],
        #     torch.tensor(self.observation["POSITION"])[self.cfg.action_indices],
        #     torch.tensor(self.observation["VELOCITY"])[self.cfg.action_indices],
        #     torch.tensor([self.mode, self.command_velocity_x, self.command_velocity_y, self.command_velocity_yaw],
        #                 dtype=torch.float32),
        # ], dim=-1)
    
    def calibration_robot(self):
        self._set_calibration_mode(self.HUMANOID_LITE_JOINTS_INFO["arm_left_shoulder_roll_joint"][0], 
                                   self.HUMANOID_LITE_JOINTS_INFO["arm_left_shoulder_roll_joint"][1])
        self._set_calibration_mode(self.HUMANOID_LITE_JOINTS_INFO["arm_right_shoulder_roll_joint"][0], 
                                   self.HUMANOID_LITE_JOINTS_INFO["arm_right_shoulder_roll_joint"][1])
        time.sleep(20)


        self._set_pos_mode("arm_left_shoulder_roll_joint")
        self._set_pos_mode("arm_right_shoulder_roll_joint")

        time.sleep(1)

        self.joint_controler.start()

        self.HUMANOID_LITE_JOINTS_GOALPOS["arm_left_shoulder_roll_joint"] = self.HUMANOID_LITE_JOINTS_INFO["arm_left_shoulder_roll_joint"][3]
        self.HUMANOID_LITE_JOINTS_GOALPOS["arm_right_shoulder_roll_joint"] = self.HUMANOID_LITE_JOINTS_INFO["arm_right_shoulder_roll_joint"][3]
        time.sleep(5)
        self.joint_tracking = False
        
        for name in self.HUMANOID_ARM_LEFT_LITE_JOINTS:
            if name == "arm_left_shoulder_roll_joint":
                continue
            self._set_calibration_mode(self.HUMANOID_LITE_JOINTS_INFO[name][0],
                                       self.HUMANOID_LITE_JOINTS_INFO[name][1],)
            
        for name in self.HUMANOID_LEG_RIGHT_LITE_JOINTS:
            self._set_calibration_mode(self.HUMANOID_LITE_JOINTS_INFO[name][0],
                                       self.HUMANOID_LITE_JOINTS_INFO[name][1],)
        

        time.sleep(15)
        
        for name in self.HUMANOID_ARM_RIGHT_LITE_JOINTS:
            if name == "arm_right_shoulder_roll_joint":
                continue
            self._set_calibration_mode(self.HUMANOID_LITE_JOINTS_INFO[name][0],
                                       self.HUMANOID_LITE_JOINTS_INFO[name][1],)

        for name in self.HUMANOID_LEG_LEFT_LITE_JOINTS:
            self._set_calibration_mode(self.HUMANOID_LITE_JOINTS_INFO[name][0],
                                       self.HUMANOID_LITE_JOINTS_INFO[name][1],)
            
            
        time.sleep(20)

        for name in self.HUMANOID_ARM_LEFT_LITE_JOINTS:
            if name == "arm_left_shoulder_roll_joint":
                continue
            self._set_pos_mode(name)
            
        for name in self.HUMANOID_LEG_RIGHT_LITE_JOINTS:
            self._set_pos_mode(name)
        
        for name in self.HUMANOID_ARM_RIGHT_LITE_JOINTS:
            if name == "arm_right_shoulder_roll_joint":
                continue
            self._set_pos_mode(name)

        for name in self.HUMANOID_LEG_LEFT_LITE_JOINTS:
            self._set_pos_mode(name)

        time.sleep(5)
        
        self.joint_tracking = True

        for name in self.HUMANOID_ARM_LEFT_LITE_JOINTS:
            if name == "arm_left_shoulder_roll_joint":
                continue
            self.HUMANOID_LITE_JOINTS_GOALPOS[name] = self.HUMANOID_LITE_JOINTS_INFO[name][3]
            
        for name in self.HUMANOID_LEG_RIGHT_LITE_JOINTS:
            self.HUMANOID_LITE_JOINTS_GOALPOS[name] = self.HUMANOID_LITE_JOINTS_INFO[name][3]
        
        for name in self.HUMANOID_ARM_RIGHT_LITE_JOINTS:
            if name == "arm_right_shoulder_roll_joint":
                continue
            self.HUMANOID_LITE_JOINTS_GOALPOS[name] = self.HUMANOID_LITE_JOINTS_INFO[name][3]

        for name in self.HUMANOID_LEG_LEFT_LITE_JOINTS:
            self.HUMANOID_LITE_JOINTS_GOALPOS[name] = self.HUMANOID_LITE_JOINTS_INFO[name][3]

        time.sleep(5)
        
        print('aaaaaaaaaaaaaaaaaaa')

        while True:
            time.sleep(0.1)

    def set_IDLE(self):
        for name in self.HUMANOID_LITE_JOINTS:
            agent, ID, _, _, _ = self.HUMANOID_LITE_JOINTS_INFO[name]
            self._set_idle_mode(agent, ID)

    def _move_motor(self, name, pos):
        agent, ID, _, _, _ = self.HUMANOID_LITE_JOINTS_INFO[name]
        measured_pos, measured_vel = agent.write_read_pdo_2(ID, pos, self.motor_param['velocity'], 0.01)
        if measured_pos != None: self.HUMANOID_LITE_JOINTS_CURRPOS[name] = pos
        return measured_pos, measured_vel
    
    def _joint_pulling(self):
        while True:
            if self.joint_tracking:
                print("joint_controll")
                self._ctrl_whole_joint_smooth()
                print("out")
    
    def _ctrl_whole_joint_smooth(self):
        try:
            vel = 5
            del_time = 0.001
            scale = 15
            step = (vel*del_time)*scale / 0.314

            obs_name = 'leg_right_ankle_roll_joint'
            # print((vel*del_time)*scale / 0.314)
            # print('tick')
            for i in range(5):
                for j in range(4):
                    ID = (j*5)+i
                    print(ID)
                    name = self.HUMANOID_LITE_JOINTS[ID]
                    pos = self.HUMANOID_LITE_JOINTS_GOALPOS[name]
                    cur = self.HUMANOID_LITE_JOINTS_CURRPOS[name]

                    distence = pos - cur
                    # print(name)
                    

                    if name == obs_name: print(name, pos, cur, distence, step)

                    print(abs(distence) > step)

                    if abs(distence) > step: 
                        if distence > step: 
                            # print(name, 'distence > step')
                            pos, vel = self._move_motor(name, cur+step)
                        else: 
                            # print(name, 'distence < step')
                            pos, vel = self._move_motor(name, cur-step)

                        if pos != None: self.observation["POSITION"][ID] = pos
                        if vel != None: self.observation["VELOCITY"][ID] = vel
                    else:
                        pos, vel = self._move_motor(name, pos)
                        if pos != None: self.observation["POSITION"][ID] = pos
                        if vel != None: self.observation["VELOCITY"][ID] = vel
                    
                    # print(name)

                    self._move_motor(name, pos)
                    time.sleep(del_time/22)
                print('aaaaaaaaa')
            print('vvv')
            
            for i in range(2):
                    ID = 20+i
                    print(ID)
                    name = self.HUMANOID_LITE_JOINTS[ID]
                    pos = self.HUMANOID_LITE_JOINTS_GOALPOS[name]
                    cur = self.HUMANOID_LITE_JOINTS_CURRPOS[name]

                    distence = pos - cur
                    
                    if name == obs_name: print(name, pos, cur, distence, step)
                    if abs(distence) > step: 
                        if distence > step: pos, vel = self._move_motor(name, cur+step)
                        else: pos, vel = self._move_motor(name, cur-step)
                        if pos != None: self.observation["POSITION"][ID] = pos
                        if vel != None: self.observation["VELOCITY"][ID] = vel

                    else:
                        pos, vel = self._move_motor(name, pos)
                        if pos != None: self.observation["POSITION"][ID] = pos
                        if vel != None: self.observation["VELOCITY"][ID] = vel

                    time.sleep(del_time/22)
        except IndexError as e:
            print(f"❌ IndexError at ID={ID}: {e}")
            print(f"   Total joints available: {len(self.HUMANOID_LITE_JOINTS)}")

        except KeyError as e:
            print(f"❌ KeyError for joint name '{name}': {e}")
            print(f"   Available joints: {list(self.HUMANOID_LITE_JOINTS_INFO.keys())}")

        except Exception as e:
            print(f"❌ Error processing joint ID={ID}, name={name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    
    def _move_smooth(self, name, pos, vel):
        next_pos = self.HUMANOID_LITE_JOINTS_CURRPOS[name]
        # print(f"next -> {next_pos}")
        distence = pos - next_pos
        step = vel / 31.4
        _time = step / distence
        # print(distence, step)
        while distence > step:
            next_pos += step
            distence -= step

            # print(distence, end=" / ")
            # print(step, end=" / ")
            # print(distence > step)

            self._move_motor(name, next_pos)
            time.sleep(0.01)
        self._move_motor(name, pos)
        time.sleep(0.01)

    def _get_imu(self):
        self.imu.flush() 
        
        line_bytes = self.imu.readline()
        
        while True:
            if not line_bytes:
                print("경고: 시리얼 타임아웃 발생, 유효 데이터 수신 실패.")
                return None, None, None 

            line_data = line_bytes.decode('utf-8').strip()

            if line_data and line_data.startswith('*'):
                break
            
            line_bytes = self.imu.readline()
        
        data = line_data.split(',')
        
        try:
            imu_data = [float(x) for x in data[1:]] 
            return[0, 0, 0, 0], [0, 0, 0], [0, 0, 0]
            return imu_data[:4], imu_data[4:7], imu_data[7:]

        except ValueError as e:
            print(f"오류: IMU 데이터 파싱 실패 - {e} (원본: {line_data})")
            return None, None, None

    def _set_pos_mode(self, name):
        agent, ID, pos_start, pose_goal, _ = self.HUMANOID_LITE_JOINTS_INFO[name]
        # agent.write_position_target(ID, pos_start)
        # self._move_smooth(name, pos_start, 3)
        agent.write_position_target(ID, pos_start)
        
        self.HUMANOID_LITE_JOINTS_GOALPOS[name] = pos_start
        self.HUMANOID_LITE_JOINTS_CURRPOS[name] = pos_start
        
        agent.write_position_kp(ID, self.motor_param['kp'])
        agent.write_position_kd(ID, self.motor_param['kd'])
        agent.write_position_ki(ID, self.motor_param['ki'])
        agent.write_torque_limit(ID, self.motor_param['torque'])
        agent.write_velocity_limit(ID, 20)
        
        agent.set_mode(ID, recoil.Mode.POSITION)
        time.sleep(0.1)

    def _set_idle_mode(self, agent, ID):
        agent.set_mode(ID, recoil.Mode.IDLE)
        time.sleep(0.01)
    
    def _set_demping_mode(self, agent, ID):
        agent.set_mode(ID, recoil.Mode.DAMPING)

    def _set_calibration_mode(self, agent, ID):
        agent.set_mode(ID, recoil.Mode.CALIBRATION)

    def _feed_motor(self):
        while self.thread_run:
            for name in self.HUMANOID_LITE_JOINTS:
                agent, ID, _, _, _ = self.HUMANOID_LITE_JOINTS_INFO[name]
                agent.feed(ID)
                # print('feed')
                time.sleep(1)
            
