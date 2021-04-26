import torch

from ThirdParty.fastai_helpers import *


def _get_sfs_idxs(sizes):
    "Get the indexes of the layers where the size of the activation changes."
    feature_szs = [size[-1] for size in sizes]
    sfs_idxs = list(np.where(np.array(feature_szs[:-1]) != np.array(feature_szs[1:]))[0])
    if feature_szs[0] != feature_szs[1]: sfs_idxs = [0] + sfs_idxs
    return sfs_idxs


class SimpleUnetBlock(Module):
    """A quasi-UNet block, using `PixelShuffle_ICNR upsampling`.
       This implementation allows to pass direct feature map as an argument and doesn't require using Hooks"""

    @delegates(ConvLayer.__init__)
    def __init__(self, up_in_c, x_in_c, final_div=True, blur=False,
                 self_attention=False, init=nn.init.kaiming_normal_, norm_type=None, **kwargs):
        self.shuf = PixelShuffle_ICNR(up_in_c, up_in_c // 2, blur=blur, norm_type=norm_type)
        self.bn = batchnorm_2d(x_in_c)
        ni = up_in_c // 2 + x_in_c
        nf = ni if final_div else ni // 2
        self.conv1 = ConvLayer(ni, nf)
        self.conv2 = ConvLayer(nf, nf)
        self.relu = nn.ReLU()

    def forward(self, up_in, s):
        up_out = self.shuf(up_in)
        ssh = s.shape[-2:]
        if ssh != up_out.shape[-2:]:
            up_out = F.interpolate(up_out, s.shape[-2:], mode='nearest')
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv2(self.conv1(cat_x))


class ResFuseNet(Module):
    def __init__(self, n_in_main=3, n_in_second=1, n_classes=1, imsize=(120, 160), norm_type=None, pretrained=False,
                 blur=False,
                 blur_final=True, self_attention=False):
        self.pretrained = pretrained
        self.imsize = imsize
        # Get iterable sequential model from ResNet34 architecture
        encoder_arch = resnet34(pretrained=self.pretrained)
        encoder = nn.Sequential(*list(encoder_arch.children())[:-2])
        encoder[0] = nn.Conv2d(n_in_main, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Get layer sizes of encoder
        sfs_szs = model_sizes(encoder, size=self.imsize)
        # Get indices of layers where size of activation changes
        sfs_idxs = list(reversed(_get_sfs_idxs(sfs_szs)))
        self.sfs = hook_outputs([encoder[i] for i in sfs_idxs], detach=False)
        # Get initial conv layers for main and second encoders (cut at ReLU to hook output before maxpool)
        self.stem_main = nn.Sequential(
            nn.Conv2d(n_in_main, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True))
        self.stem_second = nn.Sequential(
            nn.Conv2d(n_in_second, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True))
        # Initial pooling layers for encoders
        self.maxpool2d_1_main = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.maxpool2d_1_second = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        # Get encoder blocks from modules of ResNet34
        encoder_blocks = [block for block in list(encoder.children()) if isinstance(block, nn.Sequential)]
        self.ec1_main, self.ec2_main, self.ec3_main, self.ec4_main = encoder_blocks
        self.ec1_second, self.ec2_second, self.ec3_second, self.ec4_second = encoder_blocks
        # ni is number of features from last block of encoder
        ni = sfs_szs[-1][1]
        # Bottleneck for main UNet
        self.middle_conv = nn.Sequential(BatchNorm(ni), nn.ReLU(),
                                         ConvLayer(ni, ni * 2, norm_type=norm_type),
                                         ConvLayer(ni * 2, ni, norm_type=norm_type))
        # Pass a dummy input through network to get sizes to construct decoder blocks
        x = dummy_eval(encoder, imsize).detach()
        x = self.middle_conv(x)
        decoder_blocks = []
        for i, idx in enumerate(sfs_idxs):
            not_final = i != len(sfs_idxs) - 1
            up_in_c, x_in_c = int(x.shape[1]), int(sfs_szs[idx][1])
            do_blur = blur and (not_final or blur_final)
            sa = self_attention and (i == len(sfs_idxs) - 3)
            unet_block = SimpleUnetBlock(up_in_c, x_in_c, final_div=not_final, blur=do_blur, self_attention=sa).eval()
            decoder_blocks.append(unet_block)
            hook_shape = self.sfs[i].stored.shape
            dummy = torch.randn(*hook_shape)
            x = unet_block(x, dummy)
        self.dec4, self.dec3, self.dec2, self.dec1 = decoder_blocks
        # ni is number of features for last layer of decoder
        ni = x.shape[1] + in_channels(encoder)
        self.head = nn.Sequential(ResBlock(1, ni, ni, stride=1, norm_type=norm_type),
                                  ConvLayer(ni, n_classes, ks=1, act_cls=None, norm_type=norm_type))

    def forward(self, main, second, skip_with_fuse=True):
        # 1st hook
        x = self.stem_main(main)
        y = self.stem_second(second)
        hook1 = x
        fuse1 = x + y

        # 2nd hook
        x = self.ec1_main(fuse1)
        y = self.ec1_second(y)
        hook2 = x
        fuse2 = x + y

        # 3rd hook
        x = self.ec2_main(fuse2)
        y = self.ec2_second(y)
        hook3 = x
        fuse3 = x + y

        # 4th hook
        x = self.ec3_main(fuse3)
        y = self.ec3_second(y)
        hook4 = x
        fuse4 = x + y

        # final encoder layer
        x = self.ec4_main(fuse4)
        y = self.ec4_second(y)
        x = x + y

        # middle
        x = self.middle_conv(x)

        # decode
        x = self.dec4(x, fuse4) if skip_with_fuse else self.dec4(x, hook4)
        x = self.dec3(x, fuse3) if skip_with_fuse else self.dec3(x, hook3)
        x = self.dec2(x, fuse2) if skip_with_fuse else self.dec2(x, hook2)
        x = self.dec1(x, fuse1) if skip_with_fuse else self.dec1(x, hook1)

        if x.shape[-2:] != main.shape[-2:]:
            x = F.interpolate(x, main.shape[-2:], mode='nearest')
        x = torch.cat([main, x], dim=1)
        x = self.head(x)
        return x


m = ResFuseNet(n_in_main=3, n_in_second=1, n_classes=1)
r = torch.randn(1, 3, 120, 160)
d = torch.randn(1, 1, 120, 160)
out = m(r, d)
print(out.shape)
