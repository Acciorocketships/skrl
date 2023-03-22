import torch

class VarianceCuriosity:

	def __init__(self, critics=[], mean=0., std=1., scale=1.):
		self.critics = critics
		self.mean = mean
		self.std = std
		self.scale = scale

	def reward(self, states, actions):
		values = [self.critics[i]({"states": states, "taken_actions": actions})[0]
				 for i in range(len(self.critics))]
		values = torch.stack(values, dim=0)
		value_var = torch.var(values, dim=0)
		scale = torch.prod(self.scaling(states), dim=-1).unsqueeze(-1)
		ri = scale * value_var
		return ri

	def scaling(self, x):
		if not isinstance(self.mean, torch.Tensor):
			self.mean = self.mean * torch.ones(x.shape[-1])
		if not isinstance(self.std, torch.Tensor):
			self.std = self.std * torch.ones(x.shape[-1])
		c = self.scale * ((2 * torch.pi) ** 0.5) * self.std
		dist = torch.distributions.Normal(loc=self.mean, scale=self.std)
		p = torch.exp(dist.log_prob(x))
		return p * c


