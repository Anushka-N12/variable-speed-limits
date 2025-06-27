class Segment:
    def __init__(self):
        self.prev_speed = None
        self.current_travel_time = 0
        self.free_flow_time = 0
        
    def process(self, raw_speed):
        if self.prev_speed is None:
            smoothed = raw_speed
        else:
            smoothed = 0.7*raw_speed + 0.3*self.prev_speed
        self.prev_speed = smoothed
        return smoothed
    
    def get_tti(self):
        # TTI calculation (in evaluation metrics)
        return (self.current_travel_time / self.free_flow_time)

def get_delay(incident):
    if incident.delay is not None:
        return incident.delay
    elif incident.magnitude == 3:
        return 300  # seconds
    elif incident.magnitude == 2:
        return 200
    elif incident.magnitude == 1:
        return 150
    
def estimate_critical_density(num_lanes, segment_length_km):
    """
    Empirical estimation:
    - Typical passenger car length: 5m + 2m gap = 7m/veh
    """
    return (1000 / (7 * num_lanes)) * segment_length_km
    