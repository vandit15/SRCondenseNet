import torch
p = torch.load("/home/cse/ug/15075036/SRCondenseNet_cpu/results/savedir/best_save_models4_2/checkpoint_025.pth.tar")
model = p['model']
print (model.state_dict())