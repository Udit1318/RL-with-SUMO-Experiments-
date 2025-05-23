# Traffic Simulation Environment Explanation

## Overview

This document explains the components of a traffic simulation environment implemented using SUMO (Simulation of Urban Mobility) for reinforcement learning. The environment models a four-way intersection controlled by traffic lights, where an agent can select from various traffic light timing patterns to optimize traffic flow.

## Environment Class Structure

The `Environment` class provides the core functionality for the traffic simulation environment:

```python
class Environment:
    def __init__(self, config, sumo_binary):
        self.sumo_binary = sumo_binary
        self.sumo_config = config
        self.action_space = self.define_action_space()
```

### Key Components

1. **Action Space Definition**
   ```python
   def define_action_space(self):
       return {
           0: (24, 8, 8, 24),   1: (24, 8, 24, 8),    2: (32, 8, 16, 8),    # ...
           # ... more action definitions ...
       }
   ```
   - Each action is represented by a tuple of 4 integers
   - These integers represent green light durations (in seconds) for each of the 4 directions at the intersection
   - The environment provides 35 different predefined timing patterns

2. **Simulation Control**
   ```python
   def start_simulation(self):
       traci.start([self.sumo_binary, "-c", self.sumo_config])

   def close_simulation(self):
       traci.close()

   def compute_waiting_time(self):
       waiting_time = {}
       controlled_lanes = traci.trafficlight.getControlledLanes("J1")
       for lane in controlled_lanes:
           waiting_time[lane] = traci.lane.getWaitingTime(lane)
       return waiting_time
   ```
   - Methods to start and close the SUMO simulation
   - Function to calculate waiting times for all lanes controlled by the traffic light

3. **Traffic Light Program Update**
   ```python
   def update_traffic_light_program(self, junction_id, green_durations):
       # ... validation checks ...
       yellow_duration = 4
       phases = []
       states = [
           "GGgrrrrrrrrr", "yyyrrrrrrrrr",
           "rrrGGgrrrrrr", "rrryyyrrrrrr",
           "rrrrrrGGgrrr", "rrrrrryyyrrr",
           "rrrrrrrrrGGg", "rrrrrrrrryyy"
       ]
       # ... create phases ...
       # ... set traffic light logic ...
       return sum(green_durations) + len(green_durations) * yellow_duration
   ```
   - Updates the traffic light timing plan based on chosen green durations
   - Adds yellow phases between green phases (4 seconds each)
   - Returns the total cycle length (sum of all green durations plus yellow phases)
   - The state strings represent traffic signal states:
     - 'G' = Green signal for a lane
     - 'g' = Green but must yield to other traffic
     - 'y' = Yellow (transition)
     - 'r' = Red

4. **State Representation**
   ```python
   def discretize_state(self, in_counts, out_counts):
       state = 0
       base = 6
       vehicle_count = in_counts - out_counts
       traffic_lenght = 4*vehicle_count
       
       # Bin traffic length into 6 categories (0-5)
       # ... binning logic ...
       
       # Convert to a single integer state representation
       state += bin_value * (base ** i)
       
       return state
   ```
   - Takes vehicle counts entering and leaving the intersection
   - Calculates net vehicle counts for each approach
   - Discretizes traffic length into 6 bins (0-5)
   - Returns a single integer representing the state using a base-6 number system

## Core Function: `apply_action_and_get_state`

This function executes a traffic light timing action, simulates a complete traffic cycle, then returns the resulting state and metrics.

### Step-by-Step Breakdown

```python
def apply_action_and_get_state(self, action):
    # 1. Get the green light durations from the action space based on the chosen action
    green_durations = self.action_space[action]
    
    # 2. Update the traffic light program and get the total cycle length
    cycle_length = self.update_traffic_light_program("J1", green_durations)
    
    # 3. Define the four edges connected to the intersection
    edge_set = ["E01", "E21", "E41", "E31"]
    
    # 4. Calculate total phase durations (green + yellow)
    phase_durations = [green_duration + 4 for green_duration in green_durations]
    
    # 5. Initialize counters for vehicles entering and leaving
    in_counts = np.zeros(4)
    out_counts = np.zeros(4)
    
    count = 0
    
    # 6. Record vehicles present on each edge before the simulation steps
    before_vehicles = {}
    for i in range(4):
        before_vehicles[i] = set(traci.edge.getLastStepVehicleIDs(edge_set[i]))

    # 7. Initialize array to store average waiting times for each edge
    avg_WTs = np.zeros(4)
    count = 0

    # 8. Run simulation steps for each phase duration
    for duration in phase_durations:
        # Run simulation steps for the duration of this phase
        for _ in duration:
            traci.simulationStep()
        
        # Calculate average waiting time for the next edge after this phase
        edgeID = edge_set[(count + 1)%4]
        avg_WTs[(count + 1)%4] = traci.edge.getWaitingTime(edgeID)/(traci.edge.getLastStepVehicleNumber(edgeID) + 1e-3)
        
        count = count + 1
        
        # After full cycle, update waiting times for remaining edges
        if count == 4:
            for seq in range(1, 4):
                avg_WTs[seq] = 0.5*(avg_WTs[seq] + traci.edge.getWaitingTime(edge_set[seq]))
    
    # 9. Record vehicles present after the simulation steps
    after_vehicles = {}
    for i in range(4):
        after_vehicles[i] = set(traci.edge.getLastStepVehicleIDs(edge_set[i]))

    # 10. Calculate cycle-wide average waiting time
    cycle_avg_WT = 0
    for i in zip(avg_WTs, phase_durations):
        cycle_avg_WT += i[0]*i[1]  # Weight by phase duration
    cycle_avg_WT /= 80  # Normalize by standard cycle length

    # 11. Calculate vehicle entry and exit counts
    for i in range(0, 4):
        in_counts[i] = len(after_vehicles[edge_set[i]] - before_vehicles[edge_set[i]])
        out_counts[i] = len(before_vehicles[edge_set[i]] - after_vehicles[edge_set[i]])

    # 12. Generate the discrete state representation
    state = self.discretize_state(in_counts, out_counts)
    
    # 13. Collect all vehicles observed during this cycle
    vehicles_seen = set()
    for i in range(4):
        vehicles_seen |= before_vehicles[i]
        vehicles_seen |= after_vehicles[i]

    # 14. Return the new state and metrics
    return state, cycle_length, cycle_avg_WT, vehicles_seen
```

### Average Waiting Time Calculation

The average waiting time calculation occurs in multiple steps:

1. **For each of the four traffic phases**:
   - Run the simulation for the phase duration
   - Measure waiting time for the next edge in sequence
   - Calculate avg waiting time = total waiting time / (number of vehicles + small epsilon to avoid division by zero)

2. **After completing the full cycle**, the function updates waiting times for edges 1-3, averaging the current value with the actual waiting time from SUMO.

3. **The cycle-wide average waiting time** is calculated as a weighted average:
   ```python
   cycle_avg_WT = 0
   for i in zip(avg_WTs, phase_durations):
       cycle_avg_WT += i[0]*i[1]  # Weight avg waiting time by phase duration
   cycle_avg_WT /= 80  # Normalize by reference cycle length
   ```

   This gives more weight to longer phases when calculating the overall average waiting time.

## Reinforcement Learning Integration

This environment is designed to work with reinforcement learning algorithms:

1. **Agent Actions**: The agent selects one of 35 possible traffic light timing patterns.

2. **Environment States**: The environment returns a discretized state representing traffic conditions.

3. **Metrics for Reward Calculation**: 
   - Cycle-wide average waiting time is the primary metric
   - Lower waiting times indicate better traffic flow, which should correspond to higher rewards

4. **Simulation Cycle**: 
   - Apply action (set traffic light timings)
   - Run simulation for one complete cycle
   - Observe results (waiting times, vehicle counts)
   - Calculate new state
   - Return information for reward calculation and next decision

5. **Goal**: The RL agent learns to select optimal traffic light timings based on current traffic conditions to minimize vehicle waiting times.