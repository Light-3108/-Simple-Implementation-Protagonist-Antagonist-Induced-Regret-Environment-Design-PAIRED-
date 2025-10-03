import torch

ant_rewards = [
    torch.tensor([[0.0],[0.7],[0.0],[0.6]]),
    torch.tensor([[0.0],[0.2],[0.0],[0.9]])
] 

prot_rewards = [
    torch.tensor([[0.6],[0.5],[0.9],[0.2]]),
    torch.tensor([[0.0],[0.0],[0.0],[0.8]])
]

adv_rewards = [
    torch.tensor([[0.0],[0.0],[0.0],[0.0]]),
    torch.tensor([[0.0],[0.0],[0.0],[0.0]])
]

ant = torch.stack([r.squeeze(-1) for r in ant_rewards])   #(T, N)
prot = torch.stack([r.squeeze(-1) for r in prot_rewards]) #(T, N)

print("Antagonist rewards:\n", ant)
print("Protagonist rewards:\n", prot)

ant_max = ant.max(dim=0).values

print(f"max: {ant_max}")
mask = prot != 0
prot_sum = prot.sum(dim=0)
prot_count = mask.sum(dim=0)
prot_mean = torch.where(prot_count > 0, prot_sum / prot_count, torch.zeros_like(prot_sum))

print(f"mean: {prot_mean}")
new_last = torch.clamp(ant_max - prot_mean, min=0).unsqueeze(-1)
print(f"new last step reward for adversary: {new_last}")
adv_rewards[-1] = new_last
print("Adversary rewards after replacement:\n", adv_rewards)
