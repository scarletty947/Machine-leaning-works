import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference 

def main():
    # Configuration for the simulation
    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)  # Initialize simulation interface

    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    # # Print initial joint angles
    # print(f"Initial joint angles: {sim.GetInitMotorAngles()}")

    # ============================================================================
    # TRUE PARAMETERS FROM URDF FILE (panda.urdf)
    # Format per link: [mass, com_x, com_y, com_z, ixx, ixy, ixz, iyy, iyz, izz]
    # ============================================================================


    # Sinusoidal reference
    # Specify different amplitude values for each joint
    amplitudes = [np.pi/4, np.pi/6, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4]  # Example amplitudes for joints
    # Specify different frequency values for each joint
    frequencies = [0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]  # Example frequencies for joints

    # Convert lists to NumPy arrays for easier manipulation in computations
    amplitude = np.array(amplitudes)
    frequency = np.array(frequencies)
    ref = SinusoidalReference(amplitude, frequency, sim.GetInitMotorAngles())  # Initialize the reference
    
    # Simulation parameters
    time_step = sim.GetTimeStep()
    current_time = 0
    max_time = 10  # seconds
    
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors
    # PD controller gains
    kp = 1000
    kd = 100

    # Initialize data storage
    tau_mes_all = []
    regressor_all = []
    
    # Part 1 specific storage
    tau_residual_link7_all = []
    regressor_link7_all = []
    regressor_1_to_6_all = []
    regressor_7_all = []
    tau_mes_all_7 =[]
    # Data collection loop
    print("\n=== Starting Data Collection ===")
    while current_time < max_time:
        # Measure current state
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        qdd_mes = sim.ComputeMotorAccelerationTMinusOne(0)
        
        # Compute sinusoidal reference trajectory
        q_d, qd_d = ref.get_values(current_time)  # Desired position and velocity
        
        # Control command
        tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_d, qd_d, kp, kd)
        cmd.SetControlCmd(tau_cmd, ["torque"]*7)  # Set the torque command
        sim.Step(cmd, "torque")

        # Get measured torque
        tau_mes = sim.GetMotorTorques(0) ##(7,)
        # print("shape of tau_mes",tau_mes.shape)
        # if dyn_model.visualizer: 
        #     for index in range(len(sim.bot)):  # Conditionally display the robot model
        #         q = sim.GetMotorAngles(index)
        #         dyn_model.DisplayModel(q)  # Update the display of the robot model

        # # Exit logic with 'q' key
        # keys = sim.GetPyBulletClient().getKeyboardEvents()
        # qKey = ord('q')
        # if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
        #     break

        # Compute regressor
        regressor = dyn_model.ComputeDynamicRegressor(q_mes, qd_mes, qdd_mes)
        Y_t = dyn_model.ComputeDynamicRegressor(q_mes,qd_mes,qdd_mes)
        Y_1_to_6 = Y_t[:, :60] #(7,60)
        Y_7 = Y_t[:, 60:] #(7,10)
        # print("Y_7 shape", Y_7.shape)
        regressor_1_to_6_all.append(Y_1_to_6)
        regressor_7_all.append(Y_7)
        regressor_all.append(Y_t)
        tau_mes_all.append(tau_mes)
        tau_mes_all_7.append(tau_mes[6])
        # # ============================================================================
        # # PART 1: Compute residual torque by subtracting known links 1-6 contribution
        # # ============================================================================
        # # Contribution from known links 1-6 (columns 0-59)
        # tau_contribution_known = regressor[:, :60] @ a_links_1_to_6_known
        
        # # Residual torque (what's left for link 7 to explain)
        # tau_residual = tau_mes.reshape(7, 1) - tau_contribution_known
        
        # # Regressor for link 7 only (columns 60-69)
        # regressor_link7 = regressor[:, 60:70]

        current_time += time_step
    regressor_stack = np.vstack(regressor_all[1000:])      # Stack all calculated regressors from list
    tau_stack = np.hstack(tau_mes_all[1000:])   
    print(tau_stack.shape)           # Stack all measured torques from list
    a_params = np.linalg.pinv(regressor_stack) @ tau_stack # Compute dynamic parameters
    print("a_params",a_params)
    # Stack the decoupled regressors
    regressor_Y_7_stack = np.vstack(regressor_7_all[1000:])
    print("shape of regressor_Y_7_stack",regressor_Y_7_stack.shape )
    regressor_Y_1_to_6_stack = np.vstack(regressor_1_to_6_all[1000:])

    # Extract actual dynamic a parameters from the urdf file
    a_params_1_to_6 = []
    for i in range(num_joints):
        a_params_1_to_6.append(np.asarray(dyn_model.pin_model.inertias[i+1].toDynamicParameters(), dtype=float).flatten())
    a_params_1_to_6 = np.asarray(a_params_1_to_6).flatten()
    a_params_1_to_6 = a_params_1_to_6[:10 * (num_joints - 1)]

    tau_7 = tau_stack - (regressor_Y_1_to_6_stack @ a_params_1_to_6)
    print(tau_7)
    tau_7_direct =  np.hstack(tau_mes_all_7[1000:]) 
    print(tau_7_direct)
    print((regressor_Y_1_to_6_stack @ a_params_1_to_6).shape) 
    print(np.array_equal(tau_7, tau_7_direct)) 
    # Compute and print dynamic parameters for link 7
    a_7_params = np.linalg.pinv(regressor_Y_7_stack) @ (tau_stack - (regressor_Y_1_to_6_stack @ a_params_1_to_6))
    print("a_7_params",a_7_params)
    # print("\n=== Data Collection Complete ===")
    # print(f"regressor_all shape: {regressor_all.shape}")
    # print(f"tau_mes_all shape: {tau_mes_all.shape}")
    # print(f"regressor_link7_all shape: {regressor_link7_all.shape}")
    # print(f"tau_residual_link7_all shape: {tau_residual_link7_all.shape}")

    # print("\n" + "="*60)
    # print("DIAGNOSTIC ANALYSIS")
    # print("="*60)
    
    # # 1. Check torques
    # print("\n=== Torque Analysis ===")
    # print(f"Shape: {tau_mes_all.shape}")
    # print(f"Min: {tau_mes_all.min():.4f}, Max: {tau_mes_all.max():.4f}")
    # print(f"Mean: {tau_mes_all.mean():.4f}, Std: {tau_mes_all.std():.4f}")
    # print(f"First 5 measurements (all joints):\n{tau_mes_all[:5]}")
    # print(f"Are all torques zero? {np.allclose(tau_mes_all, 0)}")
    
    # # 2. Check regressor
    # print("\n=== Regressor Analysis ===")
    # print(f"Shape: {regressor_all.shape}")
    # print(f"Min: {regressor_all.min():.4e}, Max: {regressor_all.max():.4e}")
    # print(f"Mean: {regressor_all.mean():.4e}, Std: {regressor_all.std():.4e}")
    
    # # Check for zero columns
    # col_norms = np.linalg.norm(regressor_all, axis=0)
    # near_zero_cols = np.where(col_norms < 1e-10)[0]
    # print(f"Number of near-zero columns: {len(near_zero_cols)}")
    # if len(near_zero_cols) > 0:
    #     print(f"Near-zero column indices: {near_zero_cols}")
    
    # # 3. Check conditioning
    # print("\n=== Matrix Conditioning ===")
    # try:
    #     U, s, Vt = np.linalg.svd(regressor_all, full_matrices=False)
    #     print(f"Largest singular value: {s[0]:.4e}")
    #     print(f"Smallest singular value: {s[-1]:.4e}")
    #     print(f"Condition number: {s[0]/s[-1]:.4e}")
    #     print(f"Singular values < 1e-10: {np.sum(s < 1e-10)}")
    #     print(f"First 10 singular values: {s[:10]}")
    #     print(f"Last 10 singular values: {s[-10:]}")
    # except Exception as e:
    #     print(f"SVD failed: {e}")
    
    # # 4. Try solving the system
    # print("\n=== Attempting Regression ===")
    # try:
    #     a_all = np.linalg.pinv(regressor_all) @ tau_mes_all
    #     print(f"Regression successful, shape: {a_all.shape}")
    #     print(f"Parameter range: [{a_all.min():.4f}, {a_all.max():.4f}]")
    #     print(f"First 10 parameters: {a_all[:10].flatten()}")
    #     print(f"Reshaped (7x10):\n{a_all.reshape(7, 10)}")
        
    #     # Check prediction error
    #     tau_pred = regressor_all @ a_all
    #     error = tau_mes_all - tau_pred
    #     print(f"\nPrediction error - Mean: {error.mean():.4e}, Std: {error.std():.4e}")
    #     print(f"Relative error: {np.linalg.norm(error) / np.linalg.norm(tau_mes_all):.4e}")
    # except Exception as e:
    #     print(f"Regression failed: {e}")
    
    # print("\n" + "="*60)
    
    # # ============================================================================
    # # PART 2: Estimate all parameters (all 7 links)
    # # ============================================================================
    # print("\n=== Part 2: Estimating all parameters ===")
    # a_all = np.linalg.pinv(regressor_all) @ tau_mes_all
    # print(f"Estimated parameters shape: {a_all.shape}")
    # print(f"Estimated parameters (reshaped 7x10):\n{a_all.reshape(7, 10)}")
    
    # # Prediction for Part 2
    # tau_mes_all_pred = regressor_all @ a_all
    
    # # After Part 2 regression:
    # print("\n=== Part 2: Estimating all parameters ===")
    # a_all_estimated = np.linalg.pinv(regressor_all) @ tau_mes_all

    # # Use Part 2's results as "known" for Part 1
    # a_links_1_to_6_known = a_all_estimated[0:60]
    # a_link7_reference = a_all_estimated[60:70]

    # print(f"Using Part 2's link 7 as reference: {a_link7_reference.T}")

    # # Now Part 1
    # regressor_link7_all = regressor_all[:, 60:70]
    # tau_contribution_known = regressor_all[:, 0:60] @ a_links_1_to_6_known
    # tau_residual_link7_all = tau_mes_all - tau_contribution_known

    # a_link7_part1 = np.linalg.pinv(regressor_link7_all) @ tau_residual_link7_all

    # print(f"\n=== Part 1 Results ===")
    # print(f"Link 7 from Part 2: {a_link7_reference.T}")
    # print(f"Link 7 from Part 1: {a_link7_part1.T}")
    # print(f"Difference: {(a_link7_part1 - a_link7_reference).T}")
    # print(f"Max absolute difference: {np.max(np.abs(a_link7_part1 - a_link7_reference)):.6f}")
        
    # # ============================================================================
    # # PART 1: Estimate only link 7 parameters using residual torque
    # # ============================================================================
    # print("\n=== Part 1: Estimating link 7 parameters only ===")
    # a_link7 = np.linalg.pinv(regressor_link7_all) @ tau_residual_link7_all
    # print(f"Estimated link 7 parameters shape: {a_link7.shape}")
    # print(f"Estimated link 7 parameters:\n{a_link7.T}")
    # print(f"True link 7 parameters:\n{a_link7_true.T}")
    # print(f"Parameter errors:\n{(a_link7 - a_link7_true).T}")
    
    # # Prediction for Part 1
    # # To get full torque prediction, we need to add back the known contribution
    # tau_mes_link7_pred = regressor_link7_all @ a_link7 + (regressor_all[:, :60] @ a_links_1_to_6_known)

    # # ============================================================================
    # # Save data
    # # ============================================================================
    # np.savez('./robot_data.npz', 
    #          # Part 1 data
    #          Y_link7=regressor_link7_all, 
    #          a_link7=a_link7,
    #          a_link7_true=a_link7_true,
    #          u_link7=tau_mes_all,  # Full measured torque for comparison
    #          u_link7_pred=tau_mes_link7_pred,  # Full predicted torque
    #          u_link7_residual=tau_residual_link7_all,  # Residual torque used for regression
    #          # Part 2 data
    #          Y=regressor_all, 
    #          a=a_all,
    #          a_true=a_true.reshape(70, 1),
    #          u=tau_mes_all,
    #          u_pred=tau_mes_all_pred)
    
    # print("\n=== Data saved to robot_data.npz ===")
    # print("Saved variables:")
    # print("  Part 1: Y_link7, a_link7, a_link7_true, u_link7, u_link7_pred, u_link7_residual")
    # print("  Part 2: Y, a, a_true, u, u_pred")

if __name__ == '__main__':
    main()