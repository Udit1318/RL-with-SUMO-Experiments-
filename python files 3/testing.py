import traci
import torch
import random


def run_simulation_test(environment, agent, episodes, episode_length):
    total_waiting_times_rl = []
    waiting_time_per_cycle_rl = []
    waiting_times_per_vehicle_rl = []
    total_vehicles_list = [] 

    print("\nEpisode Results")
    print("Format: Episode | Total Wait Time | Vehicles | Wait Time per Cycle | Wait Time per Vehicle")
    print("-" * 88)

    for episode in range(episodes):
        # Start the simulation
        environment.start_simulation()

        # Initialize episode metrics
        total_waiting_time = 0.0
        vehicle_counts = set()

        # Initialize state tracking
        step = 0
        previous_avg_waiting_time = 0
        cycle_waiting_time_accumulator = 0
        cycle_waiting_time_accumulator_PV = 0
        current_state = 0

        while step < episode_length:
            # One-hot encode the current state
            state_one_hot = [0] * agent.num_states
            state_one_hot[current_state] = 1

            # Choose action using the fixed policy
            action, action_prob = agent.choose_action(state_one_hot)

            # Apply action and get next state info
            next_state, needed_steps, cycle_avg_waiting_time, cycle_WT_per_vehicle, vehicles_seen = environment.apply_action_and_get_state(action)

            # Accumulate waiting times
            cycle_waiting_time_accumulator += cycle_avg_waiting_time
            cycle_waiting_time_accumulator_PV += cycle_WT_per_vehicle
            vehicle_counts |= vehicles_seen

            # Measure current waiting time
            current_waiting_time = sum(environment.compute_waiting_time().values())
            total_waiting_time += current_waiting_time

            # Update state and bookkeeping
            current_state = next_state
            previous_avg_waiting_time = cycle_avg_waiting_time
            step += needed_steps

        # Compute cycles and per-vehicle wait
        total_cycles = episode_length / 80
        total_vehicles = len(vehicle_counts)
        total_vehicles_list.append(total_vehicles)
        waiting_time_per_cycle = cycle_waiting_time_accumulator / total_cycles
        waiting_time_per_vehicle = cycle_waiting_time_accumulator_PV / total_cycles

        total_waiting_times_rl.append(total_waiting_time)
        waiting_time_per_cycle_rl.append(waiting_time_per_cycle)
        waiting_times_per_vehicle_rl.append(waiting_time_per_vehicle)

        # Cleanup simulation
        environment.close_simulation()

        # Print episode results
        print(f"#{episode+1:3d} | {total_waiting_time:12.1f} | {total_vehicles:3d} | {waiting_time_per_cycle:8.1f} | {waiting_time_per_vehicle:8.1f}")

    return total_waiting_times_rl, waiting_time_per_cycle_rl, waiting_times_per_vehicle_rl, total_vehicles_list
