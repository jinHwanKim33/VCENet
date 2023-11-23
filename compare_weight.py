import torch

model_1 = 'trained_models_ORG-Depth_Reward_final2/ORG-Depth_Reward_final2_TPN_nstep_22887096_2000000_2022_07_13_12_59_51.dat'
model_2 = 'trained_models_ORG-Depth_Reward_final2/ORG-Depth_Reward_final2_TPN_nstep_34246570_3000000_2022_07_13_12_59_51.dat'
saved_state_1 = torch.load(
    model_1, map_location=lambda storage, loc: storage
)
saved_state_2 = torch.load(
    model_2, map_location=lambda storage, loc: storage
)
for layers, params in saved_state_2.items():

    print(layers, torch.equal(saved_state_2[layers],saved_state_1[layers]))