import traci
import torch
import random

def run_simulation_reinforce(environment, agent, episodes, episode_length):
    total_waiting_times_rl = []
    waiting_times_per_vehicle_rl = []
    waiting_times_per_cycle_rl = []
    initial_alpha = agent.optimizer.param_groups[0]['lr']
    decay_factor = 0.01
    total_vehicles_list = []
    prev_cycle_waiting_time = 0

    print("\nEpisode Results")
    print("Format: Episode | Total Wait Time | Vehicles Waiting Time Per Cycle | Wait Time per Vehicle | Total Unique States")
    print("-" * 116)

    for episode in range(episodes):
        # Decay learning rate over time
        agent.optimizer.param_groups[0]['lr'] = initial_alpha / (1 + decay_factor * episode)
        
        # Start the simulation
        environment.start_simulation()
        
        # Initialize episode metrics
        total_waiting_time = 0.0
        vehicle_counts = set()
        unique_states = set()
        

        # Initialize trajectory storage for REINFORCE
        trajectory = []
        step = 0
        previous_avg_waiting_time = 0
        cycle_waiting_time_accumulator = 0
        vehicle_waiting_time_accumulator = 0
        current_state = 0
        
        while step < episode_length:  # Run for 1 hour simulation time
            
            # One-hot encode the current state for the agent
            state_one_hot = [0] * agent.num_states
            state_one_hot[current_state] = 1
            
            # Choose action using policy network
            action, action_prob = agent.choose_action(state_one_hot)
            
            # Apply action and get next state
            next_state, needed_steps, cycle_avg_waiting_time, cycle_WT_per_vehicle,  vehicles_seen = environment.apply_action_and_get_state(action)
            unique_states.add(next_state)

            #Adding the average waiting times of each cycles to calculate the overall avg waiting time at the end
            cycle_waiting_time_accumulator += cycle_avg_waiting_time
            vehicle_waiting_time_accumulator += cycle_WT_per_vehicle

            vehicle_counts |= vehicles_seen  # Union update

            # Measure waiting time after action
            current_waiting_time = sum(environment.compute_waiting_time().values())

            avg_waiting_time_difference = cycle_avg_waiting_time - previous_avg_waiting_time
            
            # Update metrics
            total_waiting_time += current_waiting_time
            
            
            # Compute reward (negative waiting time difference)
            # reward = -avg_waiting_time_difference
            # Bonus for reducing waiting time
            # if avg_waiting_time_difference < 0:
            #     reward = 10
            # else:
            #     reward = -4\
            # remainder = previous_avg_waiting_time - cycle_avg_waiting_time 
            # if(remainder > 0):
            #     reward = 10
            # else: reward = -1
            reward = -cycle_avg_waiting_time / 20
                
            # Add step to trajectory
            trajectory.append((state_one_hot, action, reward, action_prob))
            
            # Update state
            current_state = next_state
            previous_avg_waiting_time = cycle_avg_waiting_time
            
            # Update step count
            step += needed_steps
            
        # Update policy at the end of episode
        agent.update_policy(trajectory)
        
        #calculating the number of cycles in an episode 
        total_cycles = episode_length/80    #since the cycle lenght is fixed which is 80 seconds for each cycle

        # Calculate episode metrics
        total_vehicles = len(vehicle_counts)
        total_vehicles_list.append(total_vehicles)
        waiting_time_per_cycle = cycle_waiting_time_accumulator / total_cycles
        waiting_time_per_vehicle = vehicle_waiting_time_accumulator / total_cycles
        total_waiting_times_rl.append(total_waiting_time)
        waiting_times_per_cycle_rl.append(waiting_time_per_cycle)
        waiting_times_per_vehicle_rl.append(waiting_time_per_vehicle)
        
        # Cleanup
        environment.close_simulation()
        
        # Print episode results
        print(f"#{episode+1:3d} | {total_waiting_time:12.1f} | {total_vehicles:3d} | {waiting_time_per_cycle:8.1f} | {waiting_time_per_vehicle:8.1f} |{len(unique_states):8.1f}")
        
    return total_waiting_times_rl, waiting_times_per_cycle_rl, waiting_times_per_vehicle_rl, total_vehicles_list