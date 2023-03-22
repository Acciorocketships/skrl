import torch

class Invariance:
    def __init__(self, input_dims=slice(0,2), output_dims=slice(0,2)):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.rotation = None

    def get_rotation(self, v):
        ang = torch.atan2(v[:, 1], v[:, 0])
        rot = torch.zeros(v.shape[0], 2, 2)
        cos_ang = torch.cos(ang)
        sin_ang = torch.sin(ang)
        rot[:, 0, 0] = cos_ang
        rot[:, 0, 1] = -sin_ang
        rot[:, 1, 0] = sin_ang
        rot[:, 1, 1] = cos_ang
        self.rotation = rot

    def transform_inputs(self, x):
        x_shape = x.shape
        x = x.view(-1, x.shape[-1])
        v = x[:, self.input_dims]
        self.rotation = self.get_rotation(v)
        v_t = torch.bmm(self.rotation.permute((0,2,1)), v.unsqueeze(-1))[:,0]
        x[:,self.input_dims] = v_t
        return x.view(x_shape)

    def transform_outputs(self, x):
        x_shape = x.shape
        x = x.view(-1, x.shape[-1])
        v = x[:, self.output_dims]
        v_t = torch.bmm(self.rotation, v.unsqueeze(-1))[:, 0]
        x[:, self.output_dims] = v_t
        return x.view(x_shape)