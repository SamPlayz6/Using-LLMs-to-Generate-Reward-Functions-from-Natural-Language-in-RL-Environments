import torch
apiKey = "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"
modelName = "claude-3-5-sonnet-20240620"

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")