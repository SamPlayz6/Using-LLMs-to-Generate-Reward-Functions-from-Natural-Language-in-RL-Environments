stabilityExplanationMessage = [
        {
            "role": "user",
            "content": """Please explain the logic behind this focused stability reward function:

            def stabilityReward(observation, action):
                x, xDot, angle, angleDot = observation
                angle_stability = 1.0 - abs(angle) / 0.209
                angular_velocity_component = -abs(angleDot) / 10.0
                return float(angle_stability + angular_velocity_component)

            Explain why this function focuses specifically on stability and how each component contributes to this goal."""
        }
    ]