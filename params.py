COMBINATIONS = {
    "model": ["UNET", "UNETL"],
    "prior_provided": [False, True],
    "channels": [
        (16, 32, 64, 128, 256),
        (8, 16, 32, 64, 128),
        (4, 8, 16, 32, 64),
        (2, 4, 8, 16, 32),
    ],
    "training_data_size": [1000, 512, 256, 128, 64],
    "prior_type": ["training", "all"]
}

EXPERIMENTS = dict()
# Create a list of all possible combinations of parameters
# This is a list of dictionaries, where each dictionary is a single combination of parameters, with a name

for model in COMBINATIONS["model"]:
    for prior_provided in COMBINATIONS["prior_provided"]:
        for channels in COMBINATIONS["channels"]:
            for training_data_size in COMBINATIONS["training_data_size"]:
                for prior_type in COMBINATIONS["prior_type"]:
                    name = f"{model}_{prior_provided}_{channels[-1]}_{training_data_size}_{prior_type}"
                    EXPERIMENTS[name] = {
                        "model": model,
                        "prior_provided": prior_provided,
                        "channels": channels,
                        "training_data_size": training_data_size,
                        "prior_type": prior_type
                    }
