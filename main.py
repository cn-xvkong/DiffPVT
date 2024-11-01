class AgentAttention(nn.Module):
    def __init__(self, dim=64, num_patch=4096, num_heads=1, qkv_bias=True, qk_scale=None, attn_drop=0.1, proj_drop=0.1,
                 sr_ratio=8, agent_num=49):
        super(AgentAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        if dim == 64:
            num_heads, sr_ratio, num_patch = 1, 8, 4096
        elif dim == 128:
            num_heads, sr_ratio, num_patch = 2, 4, 1024
        elif dim == 320:
            num_heads, sr_ratio, num_patch = 5, 2, 256

        self.dim = dim
        self.num_patch = num_patch
        window_size = (int(num_patch ** 0.5), int(num_patch ** 0.5))
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.agent_num = agent_num
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window_size[0] // sr_ratio, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window_size[1] // sr_ratio))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window_size[0], 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window_size[1], agent_num))
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, H, W):
        b, n, c = x.shape
        num_heads = self.num_heads
        head_dim = c // num_heads
        q = self.q(x)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(b, c, H, W)
            x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]

        agent_tokens = self.pool(q.reshape(b, H, W, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

        kv_size = (self.window_size[0] // self.sr_ratio, self.window_size[1] // self.sr_ratio)
        position_bias1 = nn.functional.interpolate(self.an_bias, size=kv_size, mode='bilinear')
        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias = position_bias1 + position_bias2
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v

        agent_bias1 = nn.functional.interpolate(self.na_bias, size=self.window_size, mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, H // self.sr_ratio, W // self.sr_ratio, c).permute(0, 3, 1, 2)
        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v, size=(H, W), mode='bilinear')
        x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(b, n, c)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def get_sobel(in_channel, out_channel):
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)
    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_channel, axis=1)
    filter_x = np.repeat(filter_x, out_channel, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_channel, axis=1)
    filter_y = np.repeat(filter_y, out_channel, axis=0)

    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)
    conv_x = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_channel))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_channel))
    return sobel_x, sobel_y


def run_sobel(conv_x, conv_y, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
    return torch.sigmoid(g) * input


# Dual-Mode Information Filtering Module
class DMIFM(nn.Module):
    def __init__(self, in_channel):
        super(DMIFM, self).__init__()
        self.drop_out = 0.1
        self.AgentAttention = AgentAttention(dim=in_channel)
        self.sobel_x, self.sobel_y = get_sobel(in_channel=in_channel, out_channel=1)

        self.identity = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channel)
        )
        self.dropout = nn.Dropout(self.drop_out)

    def forward(self, x):
        x_attn = x
        H, W = x_attn.shape[2], x_attn.shape[3]

        x_attn = einops.rearrange(x_attn, 'B C H W -> B (H W) C')
        x_attn = self.AgentAttention(x_attn, H, W)
        x_attn = einops.rearrange(x_attn, 'B (H W) C -> B C H W', H=H, W=W)

        x_sobel = run_sobel(self.sobel_x, self.sobel_y, x)
        output = x_attn + x_sobel + self.identity(x)
        output = torch.sigmoid(output)
        output = self.dropout(output)
        return output


# Squeeze and Excitation
class SE(nn.Module):
    def __init__(self, in_channel, out_channel, ratio):
        super(SE, self).__init__()

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channel, out_channel // ratio, kernel_size=1)
        self.fc2 = nn.Conv2d(out_channel // ratio, out_channel, kernel_size=1)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x_se = self.global_avg_pool(x)
        x_se = self.relu(self.fc1(x_se))
        x_se = self.sigmoid(self.fc2(x_se))
        x = x_se * x

        return x


# Squeeze-Excitation Residual Fusion Decoder
class SERFD(nn.Module):
    def __init__(self, in_channel_high, in_channel_low, out_channel):
        super().__init__()
        in_channel_all = in_channel_low + in_channel_high
        self.drop_out = 0.1

        self.double_conv1 = nn.Sequential(
            nn.Conv2d(in_channel_all, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel)
        )

        self.double_conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel)
        )

        self.SE = SE(in_channel=out_channel, out_channel=out_channel, ratio=8)

        self.identity = nn.Sequential(
            nn.Conv2d(in_channel_all, out_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channel)
        )

        self.conv = nn.Conv2d(in_channel_all, out_channel, kernel_size=3, padding=1)
        self.BN = nn.BatchNorm2d(out_channel)
        self.ReLU = nn.ReLU(True)
        self.dropout = nn.Dropout(self.drop_out)


    def forward(self, low, high):
        while low.size()[2] != high.size()[2]:
            low = F.interpolate(low, scale_factor=2, mode='bilinear')

        fusion = torch.cat([low, high], dim=1)

        # Double_Conv add identity process fusion as a Residual function
        conv_fusion = self.double_conv1(fusion)
        SE_fusion = self.SE(conv_fusion)
        SER_fusion = self.double_conv2(SE_fusion) + self.identity(fusion)
        output = self.ReLU(SER_fusion)

        # Ablation
        # output = self.ReLU(self.BN(self.conv(fusion)))

        output = self.dropout(output)
        return output
