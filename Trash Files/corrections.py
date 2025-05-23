import traci


class Environment:
    def __init__(self, config, sumo_binary):
        self.sumo_binary = sumo_binary
        self.sumo_config = config
        self.action_space = self.define_action_space()

    def define_action_space(self):
        return {
            0: (24, 8, 8, 24),   1: (24, 8, 24, 8),    2: (32, 8, 16, 8),    3: (40, 8, 8, 8),     4: (24, 8, 16, 16),
            5: (24, 24, 8, 8),   6: (8, 32, 16, 8),    7: (16, 32, 8, 8),    8: (32, 8, 8, 16),    9: (8, 32, 8, 16),
            10: (8, 24, 24, 8),  11: (32, 16, 8, 8),   12: (8, 16, 24, 16),  13: (24, 16, 8, 16),  14: (8, 24, 16, 16),
            15: (8, 16, 8, 32),  16: (8, 16, 16, 24),  17: (8, 16, 32, 8),   18: (24, 16, 16, 8),  19: (8, 24, 8, 24),
            20: (16, 8, 24, 16), 21: (8, 40, 8, 8),    22: (8, 8, 40, 8),    23: (16, 8, 16, 24),  24: (8, 8, 32, 16),
            25: (8, 8, 16, 32),  26: (16, 8, 8, 32),   27: (8, 8, 8, 40),    28: (8, 8, 24, 24),   29: (16, 8, 32, 8),
            30: (16, 24, 16, 8), 31: (16, 16, 24, 8),  32: (16, 16, 16, 16), 33: (16, 24, 8, 16),  34: (16, 16, 8, 24)
        }

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

    def compute_queue_length(self):
        """
        Compute queue lengths for each controlled lane using the number of vehicles
        with very low speeds (i.e., effectively stopped).
        """
        queue_length = {}
        controlled_lanes = traci.trafficlight.getControlledLanes("J1")
        for lane in controlled_lanes:
            queue_length[lane] = traci.lane.getLastStepHaltingNumber(lane)
        return queue_length

    def update_traffic_light_program(self, junction_id, green_durations):
        if len(green_durations) != 4:
            raise ValueError("Must provide exactly 4 duration values")
        if not all(isinstance(x, int) and x > 0 for x in green_durations):
            raise ValueError("All durations must be positive integers")
        yellow_duration = 4
        phases = []
        states = [
            "GGgrrrrrrrrr", "yyyrrrrrrrrr",
            "rrrGGgrrrrrr", "rrryyyrrrrrr",
            "rrrrrrGGgrrr", "rrrrrryyyrrr",
            "rrrrrrrrrGGg", "rrrrrrrrryyy"
        ]
        for i, green_duration in enumerate(green_durations):
            phases.append(traci.trafficlight.Phase(
                duration=green_duration,
                state=states[i * 2],
                minDur=green_duration,
                maxDur=green_duration
            ))
            phases.append(traci.trafficlight.Phase(
                duration=yellow_duration,
                state=states[i * 2 + 1],
                minDur=yellow_duration,
                maxDur=yellow_duration
            ))
        logic = traci.trafficlight.Logic(
            programID="1",
            type=0,
            currentPhaseIndex=0,
            phases=phases
        )
        traci.trafficlight.setProgramLogic(junction_id, logic)
        return sum(green_durations) + len(green_durations) * yellow_duration  # Total cycle length

    def run_program_0(self):
        self.start_simulation()

        total_waiting_time = 0.0
        vehicle_counts = set()

        for _ in range(8400):
            traci.simulationStep()
            total_waiting_time += sum(self.compute_waiting_time().values())
            current_vehicles = set(traci.vehicle.getIDList())
            vehicle_counts.update(current_vehicles)

        total_vehicles = len(vehicle_counts)
        waiting_time_per_vehicle = total_waiting_time / total_vehicles if total_vehicles > 0 else 0
        self.close_simulation()
        print(f"\nDefault Program Results:")
        print(f"Total Wait Time: {total_waiting_time:.1f}")
        print(f"Vehicles: {total_vehicles}")
        print(f"Wait Time per Vehicle: {waiting_time_per_vehicle:.1f}")
        return total_waiting_time, total_vehicles, waiting_time_per_vehicle

    def _update_lane_flow(self, lane_id, duration):
        """
        Track vehicle flow on a lane over a specified duration.
        
        Parameters:
          lane_id (str): The ID of the lane to track
          duration (int): Number of simulation steps to track
          
        Returns:
          in_count (int): Number of vehicles that entered the lane
          out_count (int): Number of vehicles that exited the lane
        """
        # Get initial set of vehicles on the lane
        initial_vehicles = set(traci.lane.getLastStepVehicleIDs(lane_id))
        
        in_count = 0
        out_count = 0
        current_vehicles = initial_vehicles.copy()
        
        # For each step in the duration, track vehicles entering and leaving
        for _ in range(duration):
            # Get vehicles before step
            before_step = current_vehicles.copy()
            
            # Advance simulation
            traci.simulationStep()
            
            # Get vehicles after step
            after_step = set(traci.lane.getLastStepVehicleIDs(lane_id))
            
            # Vehicles that entered = in after but not before
            new_vehicles = after_step - before_step
            in_count += len(new_vehicles)
            
            # Vehicles that exited = in before but not after
            exited_vehicles = before_step - after_step
            out_count += len(exited_vehicles)
            
            # Update current vehicle set
            current_vehicles = after_step
            
        return in_count, out_count

    def apply_action_and_get_state(self, action):
        """
        Apply an action (traffic light timing) and measure the resulting flow to determine state.
        
        Parameters:
          action (int): The action index to apply
          
        Returns:
          state (int): The discretized state based on flow ratios
          needed_steps (int): Number of simulation steps that were executed
        """
        # Get the green durations for the action
        green_durations = self.action_space[action]
        
        # Update traffic light program and get the total cycle length
        cycle_length = self.update_traffic_light_program("J1", green_durations)
        
        # Get controlled lanes in order of traffic light phases
        controlled_lanes = traci.trafficlight.getControlledLanes("J1")
        
        # We need each phase's affected lanes and duration
        phase_lanes = []
        phase_durations = []
        
        # Organize controlled lanes into groups matching the green phases
        # This assumes lanes are returned in the order they appear in the phases
        phase_lanes = [
            controlled_lanes[0:3],   # Lanes for first phase
            controlled_lanes[3:6],   # Lanes for second phase
            controlled_lanes[6:9],   # Lanes for third phase
            controlled_lanes[9:12]   # Lanes for fourth phase
        ]
        
        # Each phase has green + yellow duration
        phase_durations = [green_duration + 4 for green_duration in green_durations]
        
        # Track flow ratios for each phase
        flow_ratios = []

        

        # For each phase and its corresponding lanes
        for phase_idx, (lanes, duration) in enumerate(zip(phase_lanes, phase_durations)):
            # Track total inflow and outflow for all lanes in this phase
            total_in = 0
            total_out = 0
            
            # Process each lane in the phase

            for lane in lanes:
                in_count, out_count = self._update_lane_flow(lane, duration)
                total_in += in_count
                total_out += out_count
            
            # Calculate flow ratio (with epsilon to avoid division by zero)
            
            epsilon = 1e-6
            if total_in + epsilon > 0:
                ratio = total_out / (total_in + epsilon)
            else:
                ratio = 0
                
            flow_ratios.append(ratio)
            
        # Discretize the flow ratios into state
        state = self._discretize_flow_ratios(flow_ratios)
        
        return state, cycle_length
        
    def _discretize_flow_ratios(self, flow_ratios, epsilon=1e-6):
        """
        Convert flow ratios into a discrete state.
        
        Parameters:
          flow_ratios (list): List of flow ratios for each phase
          epsilon (float): Small value to avoid division by zero
          
        Returns:
          state (int): Discretized state as an integer
        """
        state = 0
        base = 3  # Three bins: 0, 1, 2
        
        # comment : 
        # number_of_veh/

        for i, ratio in enumerate(flow_ratios):
            # Assign bin based on flow ratio
            if ratio >= 1.1:
                bin_value = 2  # Efficient clearing
            elif ratio >= 0.7:
                bin_value = 1  # Moderate clearing
            else:
                bin_value = 0  # Inefficient clearing

            #000 = 0
            #201 = 

            # Combine bins using base-3 encoding
            state += bin_value * (base ** i)
            
        return state

    def discretized_state(self, action, epsilon=1e-6):
        """
        Legacy method kept for compatibility with existing code.
        Uses apply_action_and_get_state internally.
        """
        state, _ = self.apply_action_and_get_state(action)
        return state

    def get_num_actions(self):
        return len(self.action_space)