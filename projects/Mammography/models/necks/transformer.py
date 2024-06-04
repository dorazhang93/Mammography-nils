import torch
from torch import nn , einsum
from mmengine.model import BaseModule
from mmpretrain.registry import MODELS
import torch.nn.functional as F
from einops import rearrange, repeat

class Attention(BaseModule):
    def __init__(self, dim, heads, num_patches,dim_head, dropout, is_LSA):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads ==1 and dim_head == dim)

        self.heads = heads
        self.num_patches=num_patches
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim*3, bias=False)

        self.to_out= nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        if is_LSA:
            print("is_LSA True!!!")
            self.scale = nn.Parameter(self.scale * torch.ones(heads))
            self.mask = torch.eye(self.num_patches + 1, self.num_patches + 1)
            self.mask = torch.nonzero((self.mask == 1), as_tuple=False)
        else:
            self.mask = None

    def forward(self,x):
        x=self.norm(x)
        qkv = self.to_qkv(x).chunk(3,dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        b,h,_,_ = q.shape

        if self.mask is None:
            dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        else:
            scale = self.scale
            dots = torch.mul(einsum('b h i d, b h j d -> b h i j', q, k),
                             scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((b, h, 1, 1)))
            dots[:, :, self.mask[:, 0], self.mask[:, 1]] = -987654321

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(BaseModule):
    def __init__(self, dim, hidden_dim=False, dropout=0.3):
        super(FeedForward, self).__init__()
        if hidden_dim:
            self.net = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, 1024),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(1024, dim),
                nn.Dropout(dropout)
            )
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
                nn.Dropout(dropout)
            )


    def forward(self,x):
        return self.net(x)

class Transformer(BaseModule):
    def __init__(self, dim, depth, heads,num_patches, dim_head=64,
                 ff_hidden_layer=False,
                 forward_layer=False,
                 is_LSA=False,
                 dropout=0.2):
        super(Transformer, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        self.ff_layer=forward_layer
        for _ in range(depth):
            if forward_layer:
                self.layers.append(nn.ModuleList([
                    Attention(dim, heads=heads,num_patches=num_patches, dim_head=dim_head, dropout=dropout, is_LSA=is_LSA),
                    FeedForward(dim,hidden_dim=ff_hidden_layer,dropout=0.3),
                ]))
            else:
                self.layers.append(
                    Attention(dim, heads=heads, num_patches=num_patches, dim_head=dim_head, dropout=dropout, is_LSA=is_LSA),
                )

    def forward(self,x):
        if self.ff_layer:
            for attn, ff in self.layers:
                x = attn(x)+x
                x=ff(x)+x
        else:
            for attn in self.layers:
                x = attn(x)+x
        return self.norm(x)

@MODELS.register_module()
class TransformerNeck(BaseModule):
    def __init__(self,
                 in_dims=1024,
                 out_dims=5,
                 depth=1,
                 num_heads=2,
                 head_dim=64,
                 pooling_size=4,
                 forward_layer=False,
                 ff_hidden_layer=False,
                 is_LSA=False):
        super(TransformerNeck, self).__init__()
        dim=in_dims
        if pooling_size==4:
            self.num_patches=14*8
        elif pooling_size==2:
            self.num_patches=18*16
        elif pooling_size==1:
            self.num_patches=15*15
        else:
            raise ValueError(f"Invalid pooling size {pooling_size}")
        print(f"Pooling size {pooling_size}, number of patches {self.num_patches}")

        self.pooling= nn.AvgPool2d(2,stride=1) if pooling_size==1 else nn.AvgPool2d((4,3),stride=(3,2),padding=(0,1))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(0.2)

        self.transformer = Transformer(dim,depth,num_heads,num_patches=self.num_patches,dim_head=head_dim,ff_hidden_layer=ff_hidden_layer,
                                       forward_layer=forward_layer, is_LSA=is_LSA)
        self.to_latent = nn.Identity()
        self.fc=nn.Linear(in_dims,out_dims)

    def forward(self, inputs):
        # channel first style, BCHW
        x=self.pooling(inputs[0])
        # print(x.shape)
        B,C,H,W =x.shape
        x=x.reshape(B,C,W*H).permute(0,2,1)
        b,n,_ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x=self.transformer(x)
        x=x[:,0]

        x=self.to_latent(x)
        x=self.fc(x)

        return tuple([x])


