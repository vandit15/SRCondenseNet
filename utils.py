from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable
from functools import reduce
import operator
from layers import LearnedGroupConv, CondensingLinear, CondensingConv, Conv
from torch.utils import data
from os import listdir
import torchvision.transforms as transforms
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, RandomRotation, RandomHorizontalFlip, RandomVerticalFlip
from os.path import join
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np
from numpy import newaxis
import numpy
count_ops = 0
count_params = 0


def get_num_gen(gen):
    return sum(1 for x in gen)


def is_pruned(layer):
    try:
        layer.mask
        return True
    except AttributeError:
        return False


def is_leaf(model):
    return get_num_gen(model.children()) == 0


def convert_model(model, args):
    for m in model._modules:
        child = model._modules[m]
        # if is_leaf(child):
        #     if isinstance(child, nn.Linear):
        #         model._modules[m] = CondensingLinear(child, 0.5)
        #         del(child)
        if is_pruned(child):
            model._modules[m] = CondensingConv(child)
            del(child)
        else:
            convert_model(child, args)


def get_layer_info(layer):
    layer_str = str(layer)
    type_name = layer_str[:layer_str.find('(')].strip()
    return type_name


def get_layer_param(model):
    return sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])


### The input batch size should be 1 to call this function
def measure_layer(layer, x):
    global count_ops, count_params
    delta_ops = 0
    delta_params = 0
    multi_add = 1
    type_name = get_layer_info(layer)

    ### ops_conv
    if type_name in ['Conv2d']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                    layer.stride[1] + 1)
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
        delta_params = get_layer_param(layer)

    ### ops_learned_conv
    elif type_name in ['LearnedGroupConv']:
        # measure_layer(layer.relu, x)
        # measure_layer(layer.norm, x)
        conv = layer.conv
        out_h = int((x.size()[2] + 2 * conv.padding[0] - conv.kernel_size[0]) /
                    conv.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * conv.padding[1] - conv.kernel_size[1]) /
                    conv.stride[1] + 1)
        delta_ops = conv.in_channels * conv.out_channels * conv.kernel_size[0] * \
                conv.kernel_size[1] * out_h * out_w / layer.condense_factor * multi_add
        delta_params = get_layer_param(conv) / layer.condense_factor

    ### ops_nonlinearity
    elif type_name in ['ReLU', 'LeakyReLU']:
        delta_ops = x.numel()
        delta_params = get_layer_param(layer)

    ### ops_pooling
    elif type_name in ['AvgPool2d']:
        in_w = x.size()[2]
        kernel_ops = layer.kernel_size * layer.kernel_size
        out_w = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        out_h = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        delta_ops = x.size()[0] * x.size()[1] * out_w * out_h * kernel_ops
        delta_params = get_layer_param(layer)

    elif type_name in ['AdaptiveAvgPool2d']:
        delta_ops = x.size()[0] * x.size()[1] * x.size()[2] * x.size()[3]
        delta_params = get_layer_param(layer)

    ### ops_linear
    elif type_name in ['Linear']:
        weight_ops = layer.weight.numel() * multi_add
        bias_ops = layer.bias.numel()
        delta_ops = x.size()[0] * (weight_ops + bias_ops)
        delta_params = get_layer_param(layer)

    ### ops_nothing
    elif type_name in ['BatchNorm2d', 'Dropout2d', 'DropChannel', 'Dropout']:
        delta_params = get_layer_param(layer)

    elif type_name in ['ConvTranspose2d']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                    layer.stride[1] + 1)
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
        delta_params = get_layer_param(layer)

    ### unknown layer type
    else:
        raise TypeError('unknown layer type: %s' % type_name)

    count_ops += delta_ops
    count_params += delta_params
    return


def measure_model(model, H, W, scaling_factor):
    global count_ops, count_params
    count_ops = 0
    count_params = 0
    data = Variable(torch.zeros(1, 1, H//scaling_factor, W//scaling_factor)).cuda()
    data_bicubic = Variable(torch.zeros(1, 1, H, W)).cuda()

    def should_measure(x):
        return is_leaf(x) or is_pruned(x)

    def modify_forward(model):
        for child in model.children():
            if should_measure(child):
                def new_forward(m):
                    def lambda_forward(x):
                        measure_layer(m, x)
                        return m.old_forward(x)
                    return lambda_forward
                child.old_forward = child.forward
                child.forward = new_forward(child)
            else:
                modify_forward(child)

    def restore_forward(model):
        for child in model.children():
            # leaf node
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
                child.old_forward = None
            else:
                restore_forward(child)

    modify_forward(model)
    model.forward(data,data_bicubic)
    restore_forward(model)

    return count_ops, count_params

class charbonnier_loss(nn.Module):
    """L1 Charbonnierloss. From PyTorch LapSRN"""
    def __init__(self):
        super(charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps )
        # error = diff * diff
        loss = torch.sum(error)
        return loss

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.bmp'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return transforms.Compose([
        # RandomCrop(crop_size),                                                                                           ####change 20.3
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return transforms.Compose([
        ToPILImage(),
        # Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),                                                 ####change 20.3
        Resize((crop_size[0] // upscale_factor, crop_size[1] // upscale_factor), interpolation = Image.BICUBIC),            #####change 20.3.
        ToTensor()
    ])


def display_transform():
    return transforms.Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
        ])

def Rand_Rotate_right(im):
    transform = transforms.Compose([
        RandomRotation(90),
        ])
    return transform(im)

def Rand_Rotate_left(im):
    transform =  transforms.Compose([
        RandomRotation(270),
        ])
    return transform(im)

def Rand_Flip(im):
    transform =  transforms.Compose([
        RandomHorizontalFlip(),
        ])
    return transform(im)

def Rand_Rotate_180(im):
    transform =  transforms.Compose([
        RandomRotation(180),
        ])
    return transform(im)

def Rand_Flip2(im):
    transform =  transforms.Compose([
        RandomVerticalFlip(),
        ])
    return transform(im)

def convert_rgb_to_y(image, jpeg_mode=False, max_value=255.0):
    if len(image.shape) <= 2 or image.shape[2] == 1:
        return image

    if jpeg_mode:
        xform = np.array([[0.299, 0.587, 0.114]])
        y_image = image.dot(xform.T)
    else:
        xform = np.array([[65.481 / 256.0, 128.553 / 256.0, 24.966 / 256.0]])
        y_image = image.dot(xform.T) + (16.0 * max_value / 256.0)

    return y_image


def convert_rgb_to_ycbcr(image, jpeg_mode=False, max_value=255):
    if len(image.shape) < 2 or image.shape[2] == 1:
        return image

    if jpeg_mode:
        xform = np.array([[0.299, 0.587, 0.114], [-0.169, - 0.331, 0.500], [0.500, - 0.419, - 0.081]])
        ycbcr_image = image.dot(xform.T)
        ycbcr_image[:, :, [1, 2]] += max_value / 2
    else:
        xform = np.array(
            [[65.481 / 256.0, 128.553 / 256.0, 24.966 / 256.0], [- 37.945 / 256.0, - 74.494 / 256.0, 112.439 / 256.0],
             [112.439 / 256.0, - 94.154 / 256.0, - 18.285 / 256.0]])
        ycbcr_image = image.dot(xform.T)
        ycbcr_image[:, :, 0] += (16.0 * max_value / 256.0)
        ycbcr_image[:, :, [1, 2]] += (128.0 * max_value / 256.0)

    return ycbcr_image




def resize_image_by_pil(image, scale, resampling_method="bicubic"):
    width, height = image.shape[1], image.shape[0]
    new_width = int(width * scale)
    new_height = int(height * scale)

    if resampling_method == "bicubic":
        method = Image.BICUBIC
    elif resampling_method == "bilinear":
        method = Image.BILINEAR
    elif resampling_method == "nearest":
        method = Image.NEAREST
    else:
        method = Image.LANCZOS

    if len(image.shape) == 3 and image.shape[2] == 3:
        image = Image.fromarray(image, "RGB")
        image = image.resize([new_width, new_height], resample=method)
        image = np.asarray(image)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        # the image may has an alpha channel
        image = Image.fromarray(image, "RGB")
        image = image.resize([new_width, new_height], resample=method)
        image = np.asarray(image)
    else:
        image = Image.fromarray(image.reshape(height, width))
        image = image.resize([new_width, new_height], resample=method)
        image = np.asarray(image)
        image = image.reshape(new_height, new_width, 1)
    return image

def get_ychannel(im):
    (y, Cb, Cr) = im.convert("YCbCr").split()
    return (y, Cb, Cr)

def bicubic(x, c_dim, scaling_factor):
    res = torch.Tensor(x.shape[0], c_dim, x.shape[2]*scaling_factor, x.shape[3]*scaling_factor)
    transform = transforms.ToPILImage()
    for i in range(x.shape[0]):
        p = transform(x[i].cpu())
        t = Resize((p.size[0]*scaling_factor, p.size[1]*scaling_factor), interpolation=Image.BICUBIC)
        p = t(p)
        res[i] = transforms.ToTensor()(p)
    return res

def read_data(dataset_dir, crop_size, upscale_factor, c_dim, stride):
    image_filenames =  [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    # li = torch.Tensor(6*len(image_filenames), c_dim, crop_size//upscale_factor, crop_size//upscale_factor)
    # li2 = torch.Tensor(6*len(image_filenames), c_dim, crop_size, crop_size)
    list1 = []
    list2 = []

    counter = 0
    for i in range(len(image_filenames)):
        im = Image.open(image_filenames[i])
        im_h, im_w = im.size
        (im, Cb, Cr) = get_ychannel(im)
        im_h, im_w = calculate_valid_crop_size2((im_h, im_w), upscale_factor)
        im = RandomCrop((im_w, im_h))(im)
        # im = transform(im)
        # im_rt = Rand_Rotate_right(im)
        # im_lt = Rand_Rotate_left(im)
        # im_fl = Rand_Flip(im)
        # im_180 = Rand_Rotate_180(im)
        # im_fl2 = Rand_Flip2(im)
        transform_hr = train_hr_transform(crop_size)
        transform_lr = train_lr_transform((crop_size, crop_size), upscale_factor)
        im_crop = transform_hr(im)
        # im_rt_crop = transform_hr(im_rt)
        # im_lt_crop = transform_hr(im_lt)
        # im_180_crop = transform_hr(im_180)
        # im_fl_crop = transform_hr(im_fl)
        # im_fl2_crop = transform_hr(im_fl2)
        # im_crop_low = transform_lr(im_crop)
        # im_rt_crop_low = transform_lr(im_rt_crop)
        # im_lt_crop_low = transform_lr(im_lt_crop)
        # im_180_crop_low = transform_lr(im_180_crop)
        # im_fl_crop_low = transform_lr(im_fl_crop)
        # im_fl2_crop_low = transform_lr(im_fl2_crop)
        # temp_list_low = [im_crop_low, im_rt_crop_low, im_lt_crop_low, im_fl_crop_low, im_180_crop_low, im_fl2_crop_low]
        temp_list = [im_crop]
        length = len(list1)
        for i in temp_list:
            temp = stride_data(i, stride, crop_size)
            for j in temp:
                list1.append(j)

        for i in range(len(list1) - length):
            im = transform_lr(list1[length+i])
            list2.append(im)
        # for i  in temp_list_low:
        #     temp = stride_data(i, stride//upscale_factor, crop_size//upscale_factor)
        #     for j in temp:
        #         list2.append(j) 
        # li[counter] = im_crop_low
        # li[counter+1] = im_rt_crop_low
        # li[counter+2] = im_lt_crop_low
        # li[counter+3] = im_fl_crop_low
        # li[counter+4] = im_180_crop_low
        # li[counter+5] = im_fl2_crop_low 
        # li2[counter] = im_crop
        # li2[counter+1] = im_rt_crop
        # li2[counter+2] = im_lt_crop
        # li2[counter+3] = im_fl_crop
        # li2[counter+4] = im_180_crop
        # li2[counter+5] = im_fl2_crop
        # counter += 6
    li = torch.Tensor(len(list2), c_dim, crop_size//upscale_factor, crop_size//upscale_factor)
    li2 = torch.Tensor(len(list1), c_dim, crop_size, crop_size)
    li3 = torch.Tensor(len(list2), c_dim, crop_size, crop_size)
    
    for i in list2:
        li[counter] = i
        counter += 1
    counter = 0
    for i in list1:
        li2[counter] = i
        counter += 1
    li3 = bicubic(li, c_dim, upscale_factor)
    print (len(li))
    return li, li2, li3

def stride_data(im, stride, patch):
    transform = ToPILImage()
    img = transform(im)
    im_w, im_h = img.size
    temp = []
    for x in range(0,im_w-patch+1,stride):
        for y in range(0,im_h-patch+1, stride):
            bbox = (x , y,x+ patch, y+patch)
            image_crop = img.crop(bbox)
            image_crop = ToTensor()(image_crop)
            temp.append(image_crop)
    return temp



class TrainDatasetFromFolder(data.Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        im = Image.open(self.image_filenames[index])
        hr_image = self.hr_transform(im)
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)

def calculate_valid_crop_size2(crop_size, upscale_factor):
    return (crop_size[0] - (crop_size[0] % upscale_factor), crop_size[1] - (crop_size[1] % upscale_factor))

class testDatasetFromFolder2(data.Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(testDatasetFromFolder2, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        ################## edit on 20.3        ##################
        crop_size = calculate_valid_crop_size2((hr_image.size[0], hr_image.size[1]), self.upscale_factor)
        img_read = numpy.array(hr_image)
        img_read = img_read[0:crop_size[1], 0:crop_size[0]]
        if len(img_read.shape) == 2:
            img_read2 = np.ndarray((img_read.shape[0], img_read.shape[1], 3))
            img_read2[:, :, 0] = img_read
            img_read2[:, :, 1] = img_read
            img_read2[:, :, 2] = img_read
            img_read = img_read2
        im_gt_y=convert_rgb_to_y(img_read)
        gt_yuv=convert_rgb_to_ycbcr(img_read)
        im_gt_y=im_gt_y.astype("float32")
        sc=1.0/self.upscale_factor
        img_y=resize_image_by_pil(im_gt_y,sc)
        img_hr = resize_image_by_pil(img_y, self.upscale_factor)
        return ToTensor()(img_y), ToTensor()(im_gt_y),ToTensor()(img_hr), ToTensor()(gt_yuv)
        ################### edit on 20.3   ####################

        # crop_size = calculate_valid_crop_size2((hr_image.size[0], hr_image.size[1]), self.upscale_factor)
        # lr_scale = Resize((crop_size[0] // self.upscale_factor, crop_size[1] // self.upscale_factor), interpolation=Image.BICUBIC)
        # hr_scale = Resize((crop_size[0], crop_size[1]), interpolation=Image.BICUBIC)
        # hr_image = CenterCrop(crop_size)(hr_image)
        # lr_image = lr_scale(hr_image)
        # hr_restore_img = hr_scale(lr_image)  
        # return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)


import scipy.ndimage
from numpy.ma.core import exp
from scipy.constants.constants import pi


'''
The function to compute SSIM
@param param: img_mat_1 1st 2D matrix
@param param: img_mat_2 2nd 2D matrix
'''
def compute_ssim(img_mat_1, img_mat_2):
    #Variables for Gaussian kernel definition
    img_mat_1 = img_mat_1.numpy()[0]*255
    img_mat_2 = img_mat_2.numpy()[0]*255
    gaussian_kernel_sigma=1.5
    gaussian_kernel_width=11
    gaussian_kernel=numpy.zeros((gaussian_kernel_width,gaussian_kernel_width))

    #Fill Gaussian kernel
    for i in range(gaussian_kernel_width):
        for j in range(gaussian_kernel_width):
            gaussian_kernel[i,j]=\
            (1/(2*pi*(gaussian_kernel_sigma**2)))*\
            exp(-(((i-5)**2)+((j-5)**2))/(2*(gaussian_kernel_sigma**2)))

    #Convert image matrices to double precision (like in the Matlab version)
    img_mat_1=img_mat_1.astype(numpy.float)
    img_mat_2=img_mat_2.astype(numpy.float)

    #Squares of input matrices
    img_mat_1_sq=img_mat_1**2
    img_mat_2_sq=img_mat_2**2
    img_mat_12=img_mat_1*img_mat_2

    #Means obtained by Gaussian filtering of inputs
    img_mat_mu_1=scipy.ndimage.filters.convolve(img_mat_1,gaussian_kernel)
    img_mat_mu_2=scipy.ndimage.filters.convolve(img_mat_2,gaussian_kernel)

    #Squares of means
    img_mat_mu_1_sq=img_mat_mu_1**2
    img_mat_mu_2_sq=img_mat_mu_2**2
    img_mat_mu_12=img_mat_mu_1*img_mat_mu_2

    #Variances obtained by Gaussian filtering of inputs' squares
    img_mat_sigma_1_sq=scipy.ndimage.filters.convolve(img_mat_1_sq,gaussian_kernel)
    img_mat_sigma_2_sq=scipy.ndimage.filters.convolve(img_mat_2_sq,gaussian_kernel)

    #Covariance
    img_mat_sigma_12=scipy.ndimage.filters.convolve(img_mat_12,gaussian_kernel)

    #Centered squares of variances
    img_mat_sigma_1_sq=img_mat_sigma_1_sq-img_mat_mu_1_sq
    img_mat_sigma_2_sq=img_mat_sigma_2_sq-img_mat_mu_2_sq
    img_mat_sigma_12=img_mat_sigma_12-img_mat_mu_12;

    #c1/c2 constants
    #First use: manual fitting
    c_1=6.5025
    c_2=58.5225

    #Second use: change k1,k2 & c1,c2 depend on L (width of color map)
    l=255
    k_1=0.01
    c_1=(k_1*l)**2
    k_2=0.03
    c_2=(k_2*l)**2

    #Numerator of SSIM
    num_ssim=(2*img_mat_mu_12+c_1)*(2*img_mat_sigma_12+c_2)
    #Denominator of SSIM
    den_ssim=(img_mat_mu_1_sq+img_mat_mu_2_sq+c_1)*\
    (img_mat_sigma_1_sq+img_mat_sigma_2_sq+c_2)
    #SSIM
    ssim_map=num_ssim/den_ssim
    index=numpy.average(ssim_map)

    return index

