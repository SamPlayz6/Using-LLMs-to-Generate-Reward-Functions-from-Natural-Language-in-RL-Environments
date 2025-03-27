import re
import matplotlib.pyplot as plt
import time
import base64
from collections import deque
import io
import numpy as np
import anthropic

from .rewardCodeGeneration import dynamicRewardFunction


def queryAnthropicApi(api_key, model_name, messages, max_tokens=1024):
    client = anthropic.Anthropic(api_key=api_key)
    generatedRewardFunction = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        messages=messages
    )
    return generatedRewardFunction.content[0].text

def queryAnthropicExplanation(api_key, model_name, explanation_message, max_tokens=1024):
    client = anthropic.Anthropic(api_key=api_key)
    explanationResponse = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        messages=explanation_message
    )
    return explanationResponse.content[0].text



    # ---------------------

def extractFunctionCode(responseString):
    # First try to find code between triple backticks
    code_pattern = r"```python\n(def\s+\w+\(.*?\)[\s\S]*?)\n```"
    match = re.search(code_pattern, responseString)
    
    if match:
        code = match.group(1)
        # Ensure function name matches target
        code = re.sub(r"def\s+\w+\(", "def rewardFunction", code)
        return code.strip()
        
    # Fallback to finding raw function definition
    function_pattern = r"(def\s+\w+\(.*\):[\s\S]*)"
    match = re.search(function_pattern, responseString)
    
    if not match:
        raise ValueError("No valid function definition found in the response.")

    functionString = match.group(1)
    return functionString.strip()

class RewardUpdateSystem:
    def __init__(self, apiKey: str, modelName: str, maxHistoryLength: int = 100, targetComponent: int = 1):
        self.apiKey = apiKey
        self.modelName = modelName
        self.targetComponent = targetComponent
        
        # Performance tracking
        self.rewardHistory = deque(maxlen=maxHistoryLength)
        self.episodeLengths = deque(maxlen=maxHistoryLength)
        self.episodeCount = 0
        self.lastUpdateEpisode = 0
        
        # Memory of reward functions
        self.bestRewardFunction = None
        self.bestPerformance = float('-inf')
        self.rewardFunctionHistory = []
        self.performanceHistory = []
        self.cooldownPeriod = 5000  # DRAMATICALLY INCREASED cooldown to prevent too-frequent updates
        self.evaluationWindow = 100  # Wider evaluation window for more stability
        
        # STRICT UPDATE LIMITS: This is critical to prevent too-frequent updates
        self.max_updates_per_run = 2  # Maximum number of updates in a single training run
        self.update_count = 0  # Track number of updates performed
        self.minimum_episodes_between_updates = 7500  # Force long wait between updates 
        self.update_only_at_fixed_intervals = True  # Use fixed intervals by default
        self.fixed_update_interval = 10000  # Only update at 10K episode intervals
        
        # Environment change detection
        self.environment_changes = []
        self.last_environment_state = None
        self.env_change_detected = False
        self.performance_window = deque(maxlen=200)
        self.reward_variance_window = deque(maxlen=20)
        self.last_update_performance = None  # Track performance at last update
        
        # CONSERVATIVE SAFETY LIMITS: Add fixed maximum update count
        self.absolute_max_updates = 3  # Never allow more than 3 updates total
        self.update_count_absolute = 0  # Absolute update counter
        
        # Predefined robust reward functions (fallbacks if generated ones perform poorly)
        self.stable_backup_functions = {
            # Stability component - conservative, angle-focused
            1: """def rewardFunction1(observation, action):
    x, x_dot, angle, angle_dot = observation
    
    # Primary angle-based stability reward
    angle_reward = 1.0 - (abs(angle) / 0.209)  # Normalized to [0, 1]
    
    # Penalize angular velocity to discourage oscillations
    angle_velocity_penalty = min(0.5, abs(angle_dot) / 8.0)
    
    # Small position centering component
    position_component = 0.1 * (1.0 - min(1.0, abs(x) / 2.4))
    
    # Combine with emphasis on angle stability
    reward = angle_reward - 0.7 * angle_velocity_penalty + position_component
    
    return float(reward)""",
            
            # Efficiency component - position-focused
            2: """def rewardFunction2(observation, action):
    x, x_dot, angle, angle_dot = observation
    
    # Primary position-based efficiency reward
    position_reward = 1.0 - (abs(x) / 2.4)  # Normalized to [0, 1]
    
    # Penalize high velocity for energy efficiency
    velocity_penalty = min(0.5, abs(x_dot) / 5.0)
    
    # Small angle stability component
    angle_component = 0.1 * (1.0 - abs(angle) / 0.209)
    
    # Combine with emphasis on position and energy efficiency
    reward = position_reward - 0.7 * velocity_penalty + angle_component
    
    return float(reward)"""
        }
        
        # Track and log significant environment events
        self.env_changes_log = []
        
        # This acts as a safety switch - if performance drops badly after an update,
        # we'll revert to the original function and disable further updates
        self.disable_updates_after_bad_performance = True
        self.initial_reward_function = None  # Store the initial function
        
    def generatePerformanceGraphs(self):
        """Generate graphs for both rewards and episode lengths"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot rewards
        ax1.plot(list(self.rewardHistory), label=f'Reward Component {self.targetComponent}')
        ax1.set_title('Reward History')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Reward Value')
        ax1.grid(True)
        
        # Plot episode lengths
        ax2.plot(list(self.episodeLengths), label='Episode Length', color='green')
        ax2.set_title('Episode Lengths')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps until fall')
        ax2.grid(True)
        
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(image_png).decode()


    def createUpdatePrompt(self, currentFunction: str, graph: str):
        componentType = "stability" if self.targetComponent == 1 else "efficiency"
        recentRewards = list(self.rewardHistory)[-50:] if self.rewardHistory else []
        
        # Analyze reward historical data for stability
        reward_variance = np.var(recentRewards) if len(recentRewards) > 10 else "No data"
        reward_trend = "stable" if reward_variance and isinstance(reward_variance, float) and reward_variance < 0.1 else "unstable"
        
        # Check for environment changes in the recent history
        environment_changed = False
        env_change_description = "No recent environment changes"
        if hasattr(self, 'environment_changes') and len(self.environment_changes) > 0:
            # Look for environment changes in last 1000 episodes
            recent_changes = [c for c in self.environment_changes 
                             if self.episodeCount - c['episode'] < 1000]
            if recent_changes:
                environment_changed = True
                latest_change = recent_changes[-1]
                env_change_description = f"Environment changed at episode {latest_change['episode']}: {latest_change['from']} → {latest_change['to']}"
        
        # Look at episode history to see failure patterns
        failure_patterns = "Unknown"
        if hasattr(self, 'episode_history') and len(self.episode_history) > 50:
            recent_episodes = self.episode_history[-50:]
            short_episodes = [e for e in recent_episodes if e['steps'] < 100]
            if short_episodes:
                # Extract termination reasons if available
                if 'info' in short_episodes[0] and 'termination_reason' in short_episodes[0]['info']:
                    reasons = [e['info']['termination_reason'] for e in short_episodes if 'termination_reason' in e['info']]
                    if reasons:
                        # Count occurrences
                        from collections import Counter
                        reason_counts = Counter(reasons)
                        failure_patterns = ", ".join([f"{r}: {c}" for r, c in reason_counts.most_common(2)])
        
        # Determine adaptation mode based on environment change detection
        adaptation_mode = "ADAPTIVE" if environment_changed else "STABLE"
        
        # Construct content with different prompts based on adaptation mode
        adaptive_content = f"""You are optimizing a {componentType} reward function for the CartPole reinforcement learning environment. 
        The ENVIRONMENT HAS CHANGED and the reward function needs to be adapted to maintain performance.
        
        IMPORTANT ENVIRONMENT CHANGE: {env_change_description}
        
        Create a modified version of the reward function that will work well in the new environment.
        You should adjust coefficients to prioritize {componentType} in the new environment conditions.
        
        For stability reward: Focus on maintaining the pole upright despite the changed parameters.
        For efficiency reward: Focus on centering the cart while optimizing energy usage.
        
        CONSTRAINTS:
        1. The function signature MUST remain: def rewardFunction{self.targetComponent}(observation, action)
        2. The state variables in 'observation' are:
           - observation[0]: Cart Position
           - observation[1]: Cart Velocity
           - observation[2]: Pole Angle
           - observation[3]: Pole Angular Velocity
        3. You may adjust multiple coefficients, but maintain the general structure
        4. You may make more substantial changes (5-20%) to important coefficients
        5. Do not change variable names or operation types
        
        Current Function:
        {currentFunction}
        
        Performance Metrics:
        - Average Reward: {np.mean(recentRewards) if recentRewards else 'No data'}
        - Historical Best: {self.bestPerformance if self.bestPerformance != float('-inf') else 'No data'}
        - Reward Variance: {reward_variance}
        - Reward Trend: {reward_trend}
        - Common Failure Patterns: {failure_patterns}
        
        ADAPTATION GUIDELINES:
        1. Adjust 2-3 key coefficients that impact handling of the changed environment
        2. Make changes of 5-20% to these coefficients based on failure patterns
        3. Keep the same general reward structure but adapt it to new conditions
        4. If failures are primarily position-related, adjust position weights
        5. If failures are primarily angle-related, adjust angle weights
        
        Output the adapted reward function with your changes."""
        
        stable_content = f"""Analyze this {componentType} reward component function and make a minor refinement.
        
        ENVIRONMENT IS STABLE, so make only minor refinements to the reward function.
        
        STRICT CONSTRAINTS:
        1. The function signature MUST remain: def rewardFunction{self.targetComponent}(observation, action)
        2. The state variables in 'observation' are:
           - observation[0]: Cart Position
           - observation[1]: Cart Velocity
           - observation[2]: Pole Angle
           - observation[3]: Pole Angular Velocity
        3. Do not change variable names
        4. Do not add new lines of code
        5. Do not remove any existing code
        6. Do not change any mathematical operations
        7. Only modify 1-2 numeric constants by a small amount (1-5% change)
        
        Current Function:
        {currentFunction}
        
        Performance Metrics:
        - Average Reward: {np.mean(recentRewards) if recentRewards else 'No data'}
        - Historical Best: {self.bestPerformance if self.bestPerformance != float('-inf') else 'No data'}
        - Reward Variance: {reward_variance}
        - Reward Trend: {reward_trend}
        - Common Failure Patterns: {failure_patterns}
        
        REFINEMENT GUIDELINES:
        1. Identify 1-2 coefficients for minor adjustments
        2. Make small changes (1-5%) to these coefficients
        3. Keep all variables and operations the same
        4. Keep the structure of the function identical
        
        Output the refined reward function with your minimal changes."""
        
        # Choose content based on environment change status
        content = adaptive_content if environment_changed else stable_content
        
        return [{
            "role": "user",
            "content": content
        }]
    
    def _format_performance_history(self):
        if not self.performanceHistory:
            return "No previous updates"
        
        history = "\n".join([
            f"Episode {p['episode']}: Performance {p['performance']:.2f} (Best: {p['best']:.2f})"
            for p in self.performanceHistory[-3:]  # Show last 3 updates
        ])
        return history
    
    def createCriticPrompt(self, proposedFunction: str):
        componentType = "stability" if self.targetComponent == 1 else "efficiency"
        
        return [{
            "role": "user",
            "content": f"""Act as a STRICT reward function critic for a cart-pole environment. Your goal is to ensure the proposed function makes ONLY minimal changes and follows all constraints. Analyze:
    
            {proposedFunction}
    
            STRICT REQUIREMENTS:
            1. Function name MUST be EXACTLY 'rewardFunction{self.targetComponent}'
            2. Function MUST accept observation and action parameters in that order
            3. Function MUST return a numerical reward value
            4. Function MUST be structurally almost identical to the previous version
            5. Changes MUST be limited to a single tiny coefficient adjustment (0.5-2% change maximum)
            6. NO new calculations or logic should be introduced
            7. NO variables should be renamed
            8. NO mathematical operations should be changed
    
            STRICT EVALUATION CHECKLIST:
            1. Verify function name is EXACTLY 'rewardFunction{self.targetComponent}'
            2. Verify parameters are exactly (observation, action)
            3. Verify only a SINGLE coefficient has been changed by a TINY amount (0.5-2%)
            4. Verify NO other changes have been made to the function
            5. Reject if ANY structural or logical changes were made
            6. Reject if changes exceed tiny coefficient adjustments
            
            In your analysis, identify which coefficient was changed and by what percentage. If the change is greater than 2%, reject the function.
            
            End your response with EXACTLY one line containing only "Decision: Yes" or "Decision: No".
            
            Remember: The goal is ULTRA MINIMAL changes to ensure reward function stability and avoid performance drops."""
        }]
        
    def recordEpisode(self, info, steps, totalReward):
        """Record episode information and maintain debug metrics"""
        if not hasattr(self, 'episode_history'):
            self.episode_history = []
        
        # Store episode information
        self.episode_history.append({
            'info': info,
            'steps': steps,
            'reward': totalReward,
            'episode': len(self.episode_history)
        })
        
        # Debug print every 1000 episodes
        if len(self.episode_history) % 1000 == 0:
            print(f"\nDebug Metrics at episode {len(self.episode_history)}:")
            print(f"Recent average reward: {np.mean([e['reward'] for e in self.episode_history[-100:]]):.2f}")
            print(f"Recent average steps: {np.mean([e['steps'] for e in self.episode_history[-100:]]):.2f}")
            print(f"Last update episode: {self.lastUpdateEpisode}")
        
    def validateAndUpdate(self, currentFunction: str):
        try:
            componentType = "stability" if self.targetComponent == 1 else "efficiency"
            print(f"\nBeginning restricted update for {componentType} reward function...")
            
            # Increment update counters to enforce limits
            self.update_count += 1
            self.update_count_absolute += 1
            
            # Store current function as a potential rollback point
            current_performance = np.mean(list(self.rewardHistory)[-self.evaluationWindow:]) if len(self.rewardHistory) >= self.evaluationWindow else 0
            
            # Update best historical function if current is better
            if current_performance > self.bestPerformance:
                self.bestPerformance = current_performance
                self.bestRewardFunction = currentFunction
                print(f"\nNew best historical performance: {current_performance:.4f}")
            
            # SAFETY CHECK: If this is the first update, make sure we've stored the initial function
            if not hasattr(self, 'initial_reward_function') or self.initial_reward_function is None:
                self.initial_reward_function = currentFunction
                print(f"Storing initial function as fallback")
            
            # CRITICAL PERFORMANCE CHECK: If performance dropped badly after a previous update,
            # revert to initial function and disable all future updates
            if (hasattr(self, 'last_update_performance') and 
                self.last_update_performance is not None and 
                current_performance < self.last_update_performance * 0.6):
                
                print(f"\n🚨 CRITICAL: Severe performance drop detected after previous update!")
                print(f"Previous: {self.last_update_performance:.4f}, Current: {current_performance:.4f}")
                print(f"Performance drop: {(1 - current_performance/self.last_update_performance) * 100:.1f}%")
                
                # If we have the initial function, revert to it
                if self.initial_reward_function is not None:
                    print(f"Reverting to initial reward function and disabling all future updates")
                    self.disable_all_future_updates = True
                    return self.initial_reward_function, True
                
                # Otherwise fall back to stable backup
                print(f"Reverting to stable backup function and disabling all future updates")
                self.disable_all_future_updates = True
                return self.stable_backup_functions[self.targetComponent], True
            
            # MINIMAL CHANGES ONLY: For environment changes, use simple, reliable updates
            if self.env_change_detected:
                print("\nEnvironment change detected - using special environment-adaptive update approach")
                
                # For environment changes, allow slightly larger modifications but less complex ones
                graph = self.generatePerformanceGraphs()
                updatePrompt = self.createUpdatePrompt(currentFunction, graph)
                proposedFunction = queryAnthropicApi(self.apiKey, self.modelName, updatePrompt)
                newFunctionCode = extractFunctionCode(proposedFunction)
                
                # Use a directed blend that focuses on handling the environment change
                # but still maintains a strong connection to the original function
                blend_ratio = 0.15  # Allow 15% influence from new function
                
                # Create blended function with parameter emphasis for env changes
                blended_function = self._create_blended_function(currentFunction, newFunctionCode, blend_ratio)
                print("\nCreated environment-adaptive blended function")
                
                # Update tracking
                self.last_update_performance = current_performance
                
                # Record function for history
                self.rewardFunctionHistory.append({
                    'episode': self.episodeCount,
                    'function': blended_function,
                    'performance': current_performance,
                    'trigger': 'environment_change'
                })
                
                return blended_function, True
                
            # STANDARD FIXED-INTERVAL UPDATE: Extremely small, controlled changes
            print("\nPerforming standard fixed-interval update with minimal changes")
            
            # Store current performance for comparison after update
            self.last_update_performance = current_performance
            
            # Generate ultra-conservative update prompt
            graph = self.generatePerformanceGraphs()
            updatePrompt = self.createUpdatePrompt(currentFunction, graph)
            
            # Get proposed function from API
            proposedFunction = queryAnthropicApi(self.apiKey, self.modelName, updatePrompt)
            newFunctionCode = extractFunctionCode(proposedFunction)
            
            # Validate proposed function
            if self._has_extreme_coefficients(newFunctionCode):
                print("\nRejecting proposed function due to extreme coefficients")
                # Increment counters but keep current function
                return currentFunction, False
                
            # ULTRA-MINIMAL CHANGES: Tiny blend ratio for standard updates
            blend_ratio = 0.02  # Only 2% influence from new function
            
            # Create blended function with microscopic changes
            blended_function = self._create_blended_function(currentFunction, newFunctionCode, blend_ratio)
            print("\nCreated ultra-minimal blended function")
            
            # Record function for history
            self.rewardFunctionHistory.append({
                'episode': self.episodeCount,
                'function': blended_function,
                'performance': current_performance,
                'trigger': 'fixed_interval'
            })
            
            return blended_function, True
                
        except Exception as e:
            print(f"\nError during update: {e}")
            
            # Still count this as an update attempt
            if not hasattr(self, 'update_count'):
                self.update_count = 1
            if not hasattr(self, 'update_count_absolute'):
                self.update_count_absolute = 1
                
            # Return current function for safety
            print("\nError occurred - keeping current function")
            return currentFunction, False
            
    def _create_blended_function(self, old_function, new_function, blend_ratio=0.05):
        """
        Create a blended function for adaptive reward learning.
        Balances adaptation with stability by using weighted blending.
        
        Args:
            old_function: Current reward function code
            new_function: Proposed new reward function code
            blend_ratio: How much of the new function to blend in (default: 0.05 - 5%)
        """
        import re
        
        try:
            # Adapt blend ratio based on environment change detection
            if hasattr(self, 'env_change_detected') and self.env_change_detected:
                # More aggressive adaptation when environment changes are detected
                blend_ratio = min(0.3, blend_ratio * 3)  # Up to 30% of new function
                print(f"Environment change detected - using higher blend ratio: {blend_ratio:.2f}")
                self.env_change_detected = False  # Reset flag after use
            
            # First, identify magic numbers in both functions
            old_nums = re.findall(r'([-+]?\d*\.\d+|\d+)', old_function)
            new_nums = re.findall(r'([-+]?\d*\.\d+|\d+)', new_function)
            
            # If different number of coefficients, use the safer approach - keep old function
            if len(old_nums) != len(new_nums) or len(old_nums) == 0:
                print("Function structure changed - keeping original function with new name only")
                func_name_pattern = r'def\s+(\w+)\s*\('
                old_name = re.search(func_name_pattern, old_function).group(1)
                new_name = re.search(func_name_pattern, new_function).group(1)
                
                # Simply rename the old function to match expected name
                return old_function.replace(f"def {old_name}", f"def {new_name}")
            
            # Extract all numeric values with their positions in the code
            old_nums_with_pos = [(m.group(), m.start()) for m in re.finditer(r'[-+]?\d*\.\d+|\d+', old_function)]
            new_nums_with_pos = [(m.group(), m.start()) for m in re.finditer(r'[-+]?\d*\.\d+|\d+', new_function)]
            
            # Find matching positions within a small tolerance
            matched_pairs = []
            for old_num, old_pos in old_nums_with_pos:
                for new_num, new_pos in new_nums_with_pos:
                    # If they're roughly at the same position in the code
                    if abs(old_pos - new_pos) < 20:  # within 20 chars is close enough
                        matched_pairs.append((old_num, new_num))
                        break
            
            # ADAPTATION IMPROVEMENT: Allow modifying multiple coefficients
            # especially when environment parameters have changed
            if matched_pairs:
                # Sort by relative change to find most important changes
                try:
                    sorted_pairs = sorted(matched_pairs, 
                                         key=lambda p: abs((float(p[1]) - float(p[0])) / float(p[0])) if float(p[0]) != 0 else float('inf'),
                                         reverse=True)
                    
                    # Select top 2-3 coefficient changes
                    # More changes allowed after environment change detection
                    num_changes = 3 if hasattr(self, 'env_change_detected') and self.env_change_detected else 2
                    selected_pairs = sorted_pairs[:min(num_changes, len(sorted_pairs))]
                except:
                    # Fall back to first few pairs if sorting fails
                    selected_pairs = matched_pairs[:min(2, len(matched_pairs))]
            else:
                selected_pairs = []
                
            print(f"Selecting {len(selected_pairs)} coefficients for adaptation")
            
            # Create a mapping of numbers to replace
            replacements = {}
            for old_val, new_val in selected_pairs:
                try:
                    old_float = float(old_val)
                    new_float = float(new_val)
                    
                    # Percentage change
                    pct_change = abs((new_float - old_float) / old_float) if old_float != 0 else float('inf')
                    
                    # Increase allowed change percentage based on environment detection
                    max_change_pct = 0.3 if hasattr(self, 'env_change_detected') and self.env_change_detected else 0.1
                    
                    # Cap changes at reasonable levels but allow more significant changes
                    # when environment appears to have changed
                    if pct_change > max_change_pct:  
                        print(f"Change limited from {pct_change*100:.2f}% to {max_change_pct*100:.2f}%")
                        # Limit change but preserve direction
                        if new_float > old_float:
                            new_float = old_float * (1 + max_change_pct)
                        else:
                            new_float = old_float * (1 - max_change_pct)
                    
                    # Calculate blended value with appropriate blend ratio
                    blended = old_float * (1-blend_ratio) + new_float * blend_ratio
                    
                    # Skip if change is too small to matter
                    if abs(blended - old_float) < 0.0001:
                        print(f"Change too small - skipping adjustment")
                        continue
                        
                    # Format to same precision as original
                    if '.' in old_val:
                        decimal_places = len(old_val.split('.')[1])
                        formatted = f"{blended:.{decimal_places}f}"
                    else:
                        formatted = str(int(round(blended)))
                    
                    print(f"Adjusting coefficient: {old_val} -> {formatted}")
                    replacements[old_val] = formatted
                    
                except Exception as e:
                    print(f"Error processing coefficient: {e}")
                    continue
            
            # Apply replacements to old function
            blended_code = old_function
            for old_val, new_val in replacements.items():
                # Only replace numbers that appear as distinct tokens
                blended_code = re.sub(r'\b' + re.escape(old_val) + r'\b', new_val, blended_code)
            
            # Ensure function name matches the new one
            func_name_pattern = r'def\s+(\w+)\s*\('
            old_name = re.search(func_name_pattern, old_function).group(1)
            new_name = re.search(func_name_pattern, new_function).group(1)
            
            if old_name != new_name:
                blended_code = blended_code.replace(f"def {old_name}", f"def {new_name}")
                
            return blended_code
            
        except Exception as e:
            print(f"Error in blending functions: {e}")
            # If blending fails, keep old function
            func_name_pattern = r'def\s+(\w+)\s*\('
            old_name = re.search(func_name_pattern, old_function).group(1)
            new_name = re.search(func_name_pattern, new_function).group(1)
            
            # Simply rename the old function to match expected name
            return old_function.replace(f"def {old_name}", f"def {new_name}")
            
    def _has_extreme_coefficients(self, function_code):
        """Check if function has extreme coefficient values that might cause instability"""
        import re
        # Look for numeric values in the code
        coefficients = re.findall(r'[-+]?\d*\.\d+|\d+', function_code)
        
        # Convert to float
        try:
            coefficients = [float(c) for c in coefficients]
            
            # Check for extreme values
            for coef in coefficients:
                if abs(coef) > 100:  # Arbitrary threshold
                    return True
                    
            return False
        except:
            # If parsing fails, assume it's safe
            return False
            
    def _functions_too_different(self, func1, func2):
        """Compare two functions to see if they are drastically different"""
        import re
        
        # Very simple implementation - just check if primary reward components look similar
        # This is a heuristic and might need refinement
        
        # Extract lines with primary reward components (e.g., angle_reward, position_reward)
        lines1 = func1.split('\n')
        lines2 = func2.split('\n')
        
        # Count significant differences
        significant_changes = 0
        
        # Check for presence of key components
        key_terms = ['angle', 'position', 'velocity', 'reward', 'penalty']
        
        for term in key_terms:
            in_func1 = any(term in line for line in lines1)
            in_func2 = any(term in line for line in lines2)
            
            if in_func1 != in_func2:
                significant_changes += 1
                
        # If too many significant structural changes, consider them too different
        return significant_changes >= 2

    def logFunctionUpdate(self, component, old_func, new_func):
        """Log when a reward function is updated"""
        print(f"\nUpdating {component} reward function:")
        print("Old function:")
        print(old_func)
        print("\nNew function:")
        print(new_func)
        print("-" * 50)

    def waitingTime(self, componentName, metrics, lastUpdateEpisode):
        """ULTRA-CONSERVATIVE waiting time function to minimize updates"""
        currentEpisode = metrics['currentEpisode']
        timeSinceUpdate = currentEpisode - lastUpdateEpisode
        
        # STRICT UPDATE LIMITING - Check absolute update count limit
        # This is a hard cap to prevent excessive updates
        if self.update_count_absolute >= self.absolute_max_updates:
            print(f"\n⛔ Maximum update count reached ({self.absolute_max_updates}). No further updates will be performed.")
            return False
            
        # First, check if updates are disabled due to bad performance after previous update
        if hasattr(self, 'disable_all_future_updates') and self.disable_all_future_updates:
            print(f"\n⛔ Updates have been disabled due to poor performance after previous update.")
            return False
        
        # FIXED INTERVAL APPROACH - Prefer predictable, limited updates
        if self.update_only_at_fixed_intervals:
            # Only update at fixed episode numbers (e.g., 10K, 20K)
            # ALSO ensure minimum spacing between updates for conservative approach
            should_update = (
                currentEpisode % self.fixed_update_interval == 0 and  # At 10K intervals
                currentEpisode > 0 and  # Not at episode 0
                timeSinceUpdate >= self.minimum_episodes_between_updates and  # Ensure minimum spacing
                self.update_count < self.max_updates_per_run  # Limit updates per run
            )
            
            # Special case: ONLY allow environment change to override fixed intervals
            # Reset env change flag
            self.env_change_detected = False
            
            # Check for environment parameter changes (like pole length)
            if 'env_params' in metrics:
                env_params = metrics['env_params']
                env_state = str(env_params)
                
                # Explicit detection of environment changes
                if self.last_environment_state is not None and env_state != self.last_environment_state:
                    print(f"\n🚨 ENVIRONMENT CHANGE DETECTED at episode {currentEpisode}")
                    print(f"Previous: {self.last_environment_state}")
                    print(f"Current: {env_state}")
                    
                    # Update environment change flag and log
                    self.env_change_detected = True
                    self.environment_changes.append({
                        'episode': currentEpisode,
                        'from': self.last_environment_state,
                        'to': env_state
                    })
                    
                    # IMPORTANT: Still enforce some minimum gap between updates (1000 episodes)
                    # This prevents overly rapid updates even on environment changes
                    if timeSinceUpdate < 1000:
                        print(f"Environment change detected, but too soon after last update ({timeSinceUpdate} episodes).")
                        print(f"Waiting for at least 1000 episodes since last update before responding to environment change.")
                        should_update = False
                    else:
                        # Only allow the update if we haven't hit our update limits
                        should_update = self.update_count < self.max_updates_per_run
                        if should_update:
                            print(f"Environment change detected - will update reward function.")
                        else:
                            print(f"Environment change detected, but update limit reached ({self.update_count}/{self.max_updates_per_run}).")
                
                # Always track the current environment state
                self.last_environment_state = env_state
                
            # If update is happening, log why
            if should_update:
                reason = "Environment Change Detected" if self.env_change_detected else f"Fixed Interval ({currentEpisode} episodes)"
                print(f"\n✅ Update scheduled at episode {currentEpisode} - Reason: {reason}")
                print(f"Update count: {self.update_count+1}/{self.max_updates_per_run}, Total updates: {self.update_count_absolute+1}/{self.absolute_max_updates}")
                
                # Save current performance for comparison after update
                recentRewards = metrics.get('recentRewards', [])
                if len(recentRewards) >= 100:
                    self.last_update_performance = np.mean(recentRewards[-100:])
                    print(f"Current performance: {self.last_update_performance:.4f}")
                
                # Store initial function for potential rollback
                if self.initial_reward_function is None and currentEpisode < 1000:
                    print(f"Storing initial reward function for possible rollback")
                    recentRewards = metrics.get('recentRewards', [])
                    if len(recentRewards) >= 50:
                        current_performance = np.mean(recentRewards[-50:])
                        self.bestPerformance = current_performance  # Set initial best performance
                    # Get the current function that was passed as metrics
                    if 'currentFunction' in metrics:
                        self.initial_reward_function = metrics['currentFunction']
                
            return should_update
            
        # === BELOW IS BACKUP LOGIC IF FIXED INTERVALS ARE DISABLED ===
        # This is much more conservative than the previous implementation
        
        # Enforce strict cooldown period - critical to prevent too-frequent updates
        if timeSinceUpdate < self.cooldownPeriod:
            return False
            
        # Check update count limit
        if self.update_count >= self.max_updates_per_run:
            print(f"\nUpdate limit reached ({self.update_count}/{self.max_updates_per_run}). No more updates will be performed.")
            return False
            
        # Require significant history before any update
        recentRewards = metrics.get('recentRewards', [])
        if len(recentRewards) < 100:  # Need substantial history
            return False
            
        # Calculate basic performance metrics
        current_performance = np.mean(recentRewards[-100:])
        
        # Only detect environment changes and severe performance degradation
        # Everything else is ignored for stability
        
        # ENVIRONMENT CHANGE DETECTION (pole length, etc)
        self.env_change_detected = False
        if 'env_params' in metrics:
            env_params = metrics['env_params']
            env_state = str(env_params)
            
            if self.last_environment_state is not None and env_state != self.last_environment_state:
                print(f"\n🚨 ENVIRONMENT CHANGE DETECTED at episode {currentEpisode}")
                self.env_change_detected = True
                self.environment_changes.append({
                    'episode': currentEpisode,
                    'from': self.last_environment_state,
                    'to': env_state
                })
            
            self.last_environment_state = env_state
        
        # PERFORMANCE MONITORING - Use very simple metrics
        # Only update if there's a massive performance drop (>50%)
        historical_best = 0
        if hasattr(self, 'bestPerformance') and self.bestPerformance != float('-inf'):
            historical_best = self.bestPerformance
        else:
            # Simple calculation of historical best
            window_size = 100
            historical_best = np.max([np.mean(recentRewards[i:i+window_size]) 
                                for i in range(0, len(recentRewards)-window_size, window_size)])
            
        # Update history for reference
        if not hasattr(self, 'performance_history_simple'):
            self.performance_history_simple = []
            
        self.performance_history_simple.append({
            'episode': currentEpisode,
            'performance': current_performance,
            'best': historical_best
        })
        
        # VERY CONSERVATIVE UPDATE TRIGGERS:
        # 1. Environment parameter change + enough time since last update
        # 2. Catastrophic performance drop (>50% below historical best)
        should_update = (
            (self.env_change_detected and timeSinceUpdate >= 1000) or  # Allow for env changes but with minimum gap
            (current_performance < 0.5 * historical_best and historical_best > 0 and timeSinceUpdate >= self.cooldownPeriod)  # Catastrophic drop only
        )
        
        if should_update:
            self.last_update_performance = current_performance
            print(f"\nConservative update trigger: {'Environment Change' if self.env_change_detected else 'Catastrophic Performance Drop'}")
            print(f"Current Performance: {current_performance:.4f}, Historical Best: {historical_best:.4f}")
            print(f"Update count: {self.update_count+1}/{self.max_updates_per_run}")
            
        return should_update


    # def waitingTime(self, componentName, metrics, lastUpdateEpisode, threshold=5000):
    #     """
    #     Determine if it's time to update the reward function
    #     """
    #     currentEpisode = metrics['currentEpisode']
    #     timeSinceUpdate = currentEpisode - lastUpdateEpisode
        
    #     # Only return True at exact intervals
    #     should_update = (currentEpisode % threshold == 0 and currentEpisode != 0)
        
    #     # Only print debug info when we're actually updating or at major intervals
    #     if should_update or currentEpisode % 2000 == 0:
    #         print(f"\nChecking for update at episode {currentEpisode}, time since last update: {timeSinceUpdate}")
        
    #     if should_update:
    #         print(f"✓ Update triggered at episode {currentEpisode}")
        
    #     return should_update