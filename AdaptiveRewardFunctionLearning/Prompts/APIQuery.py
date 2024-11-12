#API Query -------------------------------------------

import anthropic
import datetime
import json

def queryAnthropicApi(api_key, model_name, messages, max_tokens=1024):
    # Initialize the client with the given API key
    client = anthropic.Anthropic(api_key=api_key)
    
    # Generate a reward function using the provided messages
    generatedRewardFunction = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        messages=messages
    )
    
    return generatedRewardFunction.content[0].text

def queryAnthropicExplanation(api_key, model_name, explanation_message, max_tokens=1024):
    # Initialize the client with the given API key
    client = anthropic.Anthropic(api_key=api_key)
    
    # Generate explanation for the reward function based on the provided explanation message
    explanationResponse = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        messages=explanation_message
    )
    
    return explanationResponse.content[0].text


def logClaudeCall(rewardPrompt, rewardResponse, explanationPrompt, explanationResponse, logFile='claude_calls.jsonl'):
    log_entry = {
        'timestamp': datetime.datetime.now().isoformat(),
        'reward_function': {
            'prompt': rewardPrompt,
            'response': rewardResponse
        },
        'explanation': {
            'prompt': explanationPrompt,
            'response': explanationResponse
        }
    }
    
    with open(logFile, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

