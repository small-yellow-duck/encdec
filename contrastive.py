import torch
import torch.nn


class ContrastiveLoss(torch.nn.Module):
	"""
	Contrastive loss function.

	Based on:
	"""

	def __init__(self, margin=1.0):
		super(ContrastiveLoss, self).__init__()
		self.margin = margin

	def forward(self, x0, x1, y):
		# euclidian distance
		diff = x0 - x1
		dist_sq = torch.sum(torch.pow(diff, 2), 1)
		dist = torch.sqrt(dist_sq)

		mdist = self.margin - dist
		dist = torch.clamp(mdist, min=0.0)
		#loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
		loss = y * torch.sqrt(dist_sq)  + (1 - y) * dist
		#loss = torch.sum(loss) / 2.0 / x0.size()[0]
		loss = torch.mean(0.5*torch.mean(loss, 1))
		return loss



class KL(torch.nn.Module):
	"""
	Contrastive loss function.

	Based on:
	"""

	def __init__(self):
		super(KL, self).__init__()
		self.margin = margin

	def forward(self, x0, x1, y):
		# euclidian distance
		diff = x0 - x1
		dist_sq = torch.sum(torch.pow(diff, 2), 1)
		dist = torch.sqrt(dist_sq)

		mdist = self.margin - dist
		dist = torch.clamp(mdist, min=0.0)
		loss = y * torch.log(dist_sq + (1 - y) * torch.pow(dist, 2))
		loss = torch.sum(loss) / 2.0 / x0.size(0)
		return loss
		
		
class VarLoss(torch.nn.Module):
	def __init__(self):
		super(VarLoss, self).__init__()
	   
				  
	def forward(self, mu, logz):
		return torch.mean((1.0 + logz - z_mean*z_mean - torch.exp(logz)).sum(1)/2.0)
		