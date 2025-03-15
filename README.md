# C147/247 Final Project
### Winter 2025 - _Professor Jonathan Kao_

Group members: Milla Nielsen, Guanrong Xu, Tony Hancheng Mao, and Oscar Chen

Different types of Encoders we implemented are in **emg2qwerty/modules.py**

Different transform methods we implemented are in **emg2qwerty/transforms.py**

The best model hyperparameter was GRU(Hidden Size=256, Hidden Layer=5), the hop length=44, Linear_warmup_cosine_annealing, warmup_epochs=5, warmup_start_lr=1e-4, and eta_min=1e-10. 

