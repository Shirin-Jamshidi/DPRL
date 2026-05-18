config = {
    "algo": "ppo",  # change to "td3"
    "env_name": "Pendulum-v1",
    "total_steps": 200000,
    "update_freq": 2048,
    "batch_size": 256,

    "gamma": 0.99,
    "lr": 3e-4,

    "obs_dim": None,
    "act_dim": None,

    "device": "cuda"
}