# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.


import numpy as np
import torch

from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController
from berkeley_humanoid_lite.environments import MujocoSimulator, Cfg
from berkeley_humanoid_lite_lowlevel.policy.keypad import Se2Keypad
from wholebody_operator import wholebody_operator
import time


# Load configuration
cfg = Cfg.from_arguments()

if not cfg:
    raise ValueError("Failed to load config.")


# Main execution block
def main():
    """Main execution function for the MuJoCo simulation environment."""
    # Initialize environment
    robot = MujocoSimulator(cfg)
    obs = robot.reset()

    # real_robot = wholebody_operator(cfg)
    # real_robot.set_IDLE()
    # time.sleep(5)
    # real_robot.calibration_robot()
    

    # Initialize and start policy controller
    try:
        keypad = Se2Keypad()
        keypad.run()
    except IOError as e:
        print(e)
        return

    controller = RlController(cfg)
    controller.load_policy()

    # Default actions for fallback
    default_actions = np.array(cfg.default_joint_positions, dtype=np.float32)[robot.cfg.action_indices]

    # Main control loop
    try:
        while True:
            # Get command from keypad
            # robot.command_velocity = np.array([keypad.commands["velocity_x"],
            #                                    keypad.commands["velocity_y"],
            #                                    keypad.commands["velocity_yaw"]], dtype=np.float32)
            
            # robot.command_velocity = [0.0, 0.0, 0.5]

            # Send observations and receive actions
            # time.sleep(0.001)
            actions = controller.update(obs.numpy())

            # Use default actions if no actions received
            if actions is None:
                actions = default_actions

            # Execute step
            actions = torch.tensor(actions)
            # print(len(actions))
            print(actions)
            # actions[20] = 3
            # actions = torch.zeros(22)
            # actions[21] = 3.14
            obs = robot.step(actions)
            # print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
            # time.sleep(0.1)
            # real_robot.step(actions)
            # print(len(obs))
            # print(obs)
    except:
        pass
    # real_robot.set_IDLE()


if __name__ == "__main__":
    main()
