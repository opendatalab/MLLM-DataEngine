import math
from torch import nn

class LoRALinear(nn.Linear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 lora_r=8,
                 lora_alpha=16,
                 lora_dropout=0.05) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.lora_scaling = self.lora_alpha / self.lora_r

        self.lora_A = nn.Linear(in_features,
                                self.lora_r,
                                bias=False,
                                device=device,
                                dtype=dtype)
        self.lora_B = nn.Linear(self.lora_r,
                                out_features,
                                bias=False,
                                device=device,
                                dtype=dtype)

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        res = super().forward(x)
        x = x.float()
        res += self.lora_B(self.lora_A(
            self.lora_dropout(x))) * self.lora_scaling
        return res.half()
