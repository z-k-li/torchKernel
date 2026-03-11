import torch
import torch.nn as nn
import math
from torch.nn import functional as Fu

class GaussianConvolution(torch.nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, fwhm, dim=3):

        super(GaussianConvolution, self).__init__()
        # if isinstance(kernel_size, numbers.Number):
        kernel_size = [kernel_size] * dim
        # if isinstance(sigma, numbers.Number):
        fwhm = [fwhm] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        self.kernel_size = kernel_size
        sigma = []
        for i in range(len(fwhm)):
            sigma.append(fwhm[i]/2.355)
        
        kernel = deconv_kernel= 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            gauss = 1 / (std * math.sqrt(2 * math.pi)) * \
                        torch.exp(-((mgrid - mean) / std) ** 2 / 2)
            
            kernel *= gauss

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # deconv_kernel = deconv_kernel / torch.sum(deconv_kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        # deconv_kernel = kernel.permute(0,1,4,2,3)#(kernel.detach().cpu().numpy()[:,:, ::-1, ::-1, ::-1])
        # deconv_kernel = torch.from_numpy(deconv_kernel.copy())

        self.register_buffer('weight', kernel)
        # self.register_buffer('deconv_weight', deconv_kernel)

        self.groups = channels

        if dim == 1:
            self.conv = Fu.conv1d
            self.deconv = Fu.conv_transpose1d
        elif dim == 2:
            self.conv = Fu.conv2d
            self.deconv = Fu.conv_transpose2d
        elif dim == 3:
            self.conv = Fu.conv3d
            self.deconv = Fu.conv_transpose3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )
        
    def forward(self, input):
        """
        Apply gaussian filter to input using convolution.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        if((input.shape[input.dim()-1] % 2 == 0) ):          
            pad_f = pad_l = int(self.kernel_size[0]/2)           
        else:
            if (self.kernel_size[0] % 2 == 0):
                pad_f = int(self.kernel_size[0]/2-1)
            else:
                pad_f = int(self.kernel_size[0]/2)
            pad_l = int(self.kernel_size[0]/2)

        if(len(self.kernel_size)==3):
            input = Fu.pad(input, ( pad_f,pad_l, pad_f,pad_l, pad_f,pad_l), mode='reflect')
        else:
            input = Fu.pad(input, ( pad_f,pad_l, pad_f,pad_l), mode='reflect')

        return self.conv(input, weight=self.weight, padding=0, groups=self.groups)
        
    def backward(self, input):
        """
        Apply gaussian deconvolution.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """        
        pad_l = int(self.kernel_size[0]/2)

        if (self.kernel_size[0] % 2 == 0):
            if(len(self.kernel_size)==3):            
                input = Fu.pad(input, ( 1,0, 1,0, 1,0), mode='reflect')
            else:
                input = Fu.pad(input, (1,0, 1,0), mode='reflect')

        return self.deconv(input, weight=self.weight,
                        padding=(pad_l, 
                                    pad_l,
                                    pad_l),
                        groups=self.groups)
