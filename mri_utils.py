#from tensorlayer.prepro import *
import numpy as np
import skimage.measure
import scipy
#from time import localtime, strftime
#import logging


#def distort_img(x):
#    x = (x + 1.) / 2.
#    x = flip_axis(x, axis=1, is_random=True)
#    x = elastic_transform(x, alpha=255 * 3, sigma=255 * 0.10, is_random=True)
#    x = rotation(x, rg=10, is_random=True, fill_mode='constant')
#    x = shift(x, wrg=0.10, hrg=0.10, is_random=True, fill_mode='constant')
#    x = zoom(x, zoom_range=[0.90, 1.10], is_random=True, fill_mode='constant')
#    x = brightness(x, gamma=0.05, is_random=True)
#    x = x * 2 - 1
#    return x


def to_bad_img(x, mask):
    x = (x + 1.) / 2.
    fft = scipy.fftpack.fft2(x[:, :, 0])
    fft = scipy.fftpack.fftshift(fft)
    fft = fft * mask
    fft = scipy.fftpack.ifftshift(fft)
    x = scipy.fftpack.ifft2(fft)
    x = np.abs(x)
    x = x * 2 - 1
    return x[:, :, np.newaxis]

def fft_shift(x):
    fft = scipy.fftpack.fft2(x)
    fft = scipy.fftpack.fftshift(fft)
    return fft

def shift_ifft(fft):
    fft = scipy.fftpack.ifftshift(fft)
    x = scipy.fftpack.ifft2(fft)
    return x

def Downsample(x, mask):
    fft = scipy.fftpack.fft2(x)
    fft_good = scipy.fftpack.fftshift(fft)
    fft_bad = fft_good * mask
    fft = scipy.fftpack.ifftshift(fft_bad)
    x = scipy.fftpack.ifft2(fft)
    x = np.abs(x)
# before 2019.08.23
#    x = np.real(x)
    return x, fft_good, fft_bad


def SSIM(x_good, x_bad):
    assert len(x_good.shape) == 2
    ssim_res = skimage.measure.compare_ssim(x_good, x_bad)
    return ssim_res

def PSNR(x_good, x_bad):
    assert len(x_good.shape) == 2
    psnr_res = skimage.measure.compare_psnr(x_good, x_bad)
    return psnr_res

def NMSE(x_good, x_bad):
    assert len(x_good.shape) == 2
    nmse_a_0_1 = np.sum((x_good - x_bad) ** 2)
    nmse_b_0_1 = np.sum(x_good ** 2)
    # this is DAGAN implementation, which is wrong
    nmse_a_0_1, nmse_b_0_1 = np.sqrt(nmse_a_0_1), np.sqrt(nmse_b_0_1)
    nmse_0_1 = nmse_a_0_1 / nmse_b_0_1
    return nmse_0_1


def DC(x_good, x_rec, mask):
    fft_good = fft_shift(x_good)
    fft_rec = fft_shift(x_rec)
    fft = fft_good * mask + fft_rec * (1 - mask)
    x = shift_ifft(fft)
    x = np.real(x)
    return x

def fft_abs_for_map_fn(x):
    x = (x + 1.) / 2.
    x_complex = tf.complex(x, tf.zeros_like(x))[:, :, 0]
    fft = tf.spectral.fft2d(x_complex)
    fft_abs = tf.abs(fft)
    return fft_abs


def ssim(data):
    x_good, x_bad = data
    x_good = np.squeeze(x_good)
    x_bad = np.squeeze(x_bad)
    ssim_res = skimage.measure.compare_ssim(x_good, x_bad)
    return ssim_res


def psnr(data):
    x_good, x_bad = data
    psnr_res = skimage.measure.compare_psnr(x_good, x_bad)
    return psnr_res


def vgg_prepro(x):
    x = imresize(x, [244, 244], interp='bilinear', mode=None)
    x = np.tile(x, 3)
    x = x / 127.5 - 1
    return x


def logging_setup(log_dir):
    current_time_str = strftime("%Y_%m_%d_%H_%M_%S", localtime())
    log_all_filename = os.path.join(log_dir, 'log_all_{}.log'.format(current_time_str))
    log_eval_filename = os.path.join(log_dir, 'log_eval_{}.log'.format(current_time_str))

    log_all = logging.getLogger('log_all')
    log_all.setLevel(logging.DEBUG)
    log_all.addHandler(logging.FileHandler(log_all_filename))

    log_eval = logging.getLogger('log_eval')
    log_eval.setLevel(logging.INFO)
    log_eval.addHandler(logging.FileHandler(log_eval_filename))

    log_50_filename = os.path.join(log_dir, 'log_50_images_testing_{}.log'.format(current_time_str))

    log_50 = logging.getLogger('log_50')
    log_50.setLevel(logging.DEBUG)
    log_50.addHandler(logging.FileHandler(log_50_filename))

    return log_all, log_eval, log_50, log_all_filename, log_eval_filename, log_50_filename


if __name__ == "__main__":
    pass
