import torch
apiKey = "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"
modelName = "claude-3-5-sonnet-20240620"

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")