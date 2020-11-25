import torch
from pysot.core.config import cfg


def fft_prod(A, B):
    A_real = A.cpu().index_select(len(A.shape) - 1, torch.tensor([0]))
    A_img = A.cpu().index_select(len(A.shape) - 1, torch.tensor([1]))
    B_real = B.cpu().index_select(len(B.shape) - 1, torch.tensor([0]))
    B_img = B.cpu().index_select(len(B.shape) - 1, torch.tensor([1]))
    C = torch.zeros(A.shape)
    C.cpu().index_copy_(len(C.shape) - 1, torch.LongTensor([0]), A_real * B_real - A_img * B_img)
    C.cpu().index_copy_(len(C.shape) - 1, torch.LongTensor([1]), A_real * B_img + A_img * B_real)
    if cfg.CUDA:
        C = C.cuda()
    return C


def conj(A):
    A_neg = -A.cpu().index_select(len(A.shape) - 1, torch.tensor([1]))
    A.cpu().index_copy_(len(A.shape) - 1, torch.LongTensor([1]), A_neg)
    return A

def fft_adv(A, B):
    A_real = A.cpu().index_select(len(A.shape) - 1, torch.tensor([0]))
    A_img = A.cpu().index_select(len(A.shape) - 1, torch.tensor([1]))
    B_real = B.cpu().index_select(len(B.shape) - 1, torch.tensor([0]))
    B_img = B.cpu().index_select(len(B.shape) - 1, torch.tensor([1]))
    C = torch.zeros(A.shape)
    C_real_num = A_real * B_real + A_img * B_img
    C_real_den = B_real * B_real + B_img * B_img
    C.cpu().index_copy_(len(C.shape) - 1, torch.LongTensor([0]), C_real_num / C_real_den)
    C_img_num = A_img * B_real - A_real * B_img
    C_img_den = C_real_den
    C.cpu().index_copy_(len(C.shape) - 1, torch.LongTensor([1]), C_img_num / C_img_den)
    if cfg.CUDA:
        C = C.cuda()
    return C

def Trans(data, trans, lr):
    if bool(len(trans)):
        dataf = torch.rfft(data, 2, onesided=False)
        dataf_t = fft_prod(dataf, conj(trans))
        data_t = torch.irfft(dataf_t, 2, onesided=False)
        dataout = data_t * lr + data* (1 - lr)
        return dataout
    else:
        return data


def CalTrans(vars, regs, lambda_):
    varf = torch.rfft(vars, 2, onesided=False)
    regf = torch.rfft(regs, 2, onesided=False)
    kernel = fft_prod(varf, conj(varf))
    Tran = fft_adv(fft_prod(conj(varf), regf), kernel + lambda_)
    return Tran