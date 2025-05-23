import traci
import numpy as np


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
        return sum(green_durations) + len(green_durations) * yellow_duration
    

    def apply_action_and_get_state(self, action):
        green_durations = self.action_space[action]
        cycle_length = self.update_traffic_light_program("J1", green_durations)
        
        edge_set = ["E01", "E21", "E41", "E31"]
        
        phase_durations = [green_duration + 4 for green_duration in green_durations]
        
        in_counts = np.zeros(4)
        out_counts = np.zeros(4)
        
        

        before_vehicles = {}
        # Use edge ID as key
        for edge_id in edge_set:
            before_vehicles[edge_id] = set(traci.edge.getLastStepVehicleIDs(edge_id))

        avg_WTs = np.zeros(4)
        WTs = np.zeros(4)
        count = 0

        for duration in phase_durations:
            for _ in range(duration):
                traci.simulationStep()
            
            edgeID = edge_set[(count + 1)%4]
            WTs[(count + 1)%4] = traci.edge.getWaitingTime(edgeID) #cycle waiting time
            avg_WTs[(count + 1)%4] = traci.edge.getWaitingTime(edgeID)/(traci.edge.getLastStepVehicleNumber(edgeID) + 1e-3) #waiting times per vehicle
            count = count + 1
            if count == 4:
                for seq in range(1, 4):
                    WTs[seq] = (WTs[seq] + traci.edge.getWaitingTime(edge_set[seq]))
                    avg_WTs[seq] = 0.5*(avg_WTs[seq] + traci.edge.getWaitingTime(edge_set[seq])/(traci.edge.getLastStepVehicleNumber(edge_set[seq]) + 1e-3))
        # --- MODIFICATION START ---
        after_vehicles = {}
        # Use edge ID as key
        for edge_id in edge_set:
            after_vehicles[edge_id] = set(traci.edge.getLastStepVehicleIDs(edge_id))
        # --- MODIFICATION END ---

        #calculating cycle waiting time
        cycle_WT = 0
        cycle_WT = WTs.sum()


        #calculating average waiting time
        cycle_avg_WT = 0
        for i in zip(avg_WTs, phase_durations):
            cycle_avg_WT += i[0]*i[1]
        cycle_avg_WT /= 80

        #calculating in-count and out-count
        for i in range(0, 4):
            in_counts[i] = len(after_vehicles[edge_set[i]] - before_vehicles[edge_set[i]])
            out_counts[i] = len(before_vehicles[edge_set[i]] - after_vehicles[edge_set[i]])
        
        state = self.discretize_state(in_counts, out_counts)
        vehicles_seen = set()
        for i in edge_set:
            vehicles_seen |= before_vehicles[i]
            vehicles_seen |= after_vehicles[i]


            

        return state, cycle_length, cycle_WT, cycle_avg_WT, vehicles_seen
        
    def discretize_state(self, in_counts, out_counts):
        state = 0
        base = 6
        vehicle_count = in_counts - out_counts
        traffic_lenght = 4*vehicle_count
        # print(f"  Raw Counts: In={in_counts}, Out={out_counts}, Diff={traffic_lenght}")
        for i, lenght in enumerate(traffic_lenght):
            if lenght >= 120:
                bin_value = 5
            elif lenght >= 80:
                bin_value = 4
            elif lenght >= 40:
                bin_value = 3
            elif lenght >= 20:
                bin_value = 2
            elif lenght >= 10:
                bin_value = 1
            else:
                bin_value = 0

            state += bin_value * (base ** i)
            
        return state


    def get_num_actions(self):
        return len(self.action_space)