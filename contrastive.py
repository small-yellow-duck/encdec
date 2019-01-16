import torch
import torch.nn
import numpy as np



class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.

    Based on:
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.mean(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        #loss = y * torch.sqrt(dist_sq)  + (1 - y) * dist
        #loss = torch.sum(loss) / 2.0 / x0.size()[0]
        loss = torch.mean(0.5*torch.mean(loss, 1))
        return loss


class ln_gaussian_overlap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputt):
        ln_area = torch.log(1.0 - torch.erf(torch.abs(input) / 2.0))
        ctx.save_for_backward = inputt, ln_area
        return ln_area

    @staticmethod
    def backward(self, grad_output):
        inputt, ln_area = self.saved_tensors
        more_grad = 1. / math.sqrt(math.pi) * torch.exp(-(inputt ** 2)/4.0) / ln_area
        return more_grad * grad_output

class ln_1min_gaussian_overlap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputt):
        ln_area = torch.log(torch.erf(torch.abs(input) / 2.0))
        ctx.save_for_backward = inputt, ln_area
        return ln_area

    @staticmethod
    def backward(ctx, grad_output):
        inputt, ln_area = ctx.save_for_backward
        more_grad = 1. / math.sqrt(math.pi) * torch.exp(-(inputt ** 2)/4.0) / ln_area
        return more_grad * grad_output


class KL(torch.nn.Module):
    """
    Contrastive loss function.

    Based on:
    """

    def __init__(self):
        super(KL, self).__init__()

    def forward(self, mu1, mu2, y):
        #sigma1 and sigma2 are always 1

        #prod = torch.cumprod(torch.abs(torch.erf((mu2 - mu1)/2.0)), 1)[:, -1:]
        #ln_area = torch.log(torch.clamp(1.0 - prod, 1e-7, 1.0))/mu1.size(1)

        ln_area = torch.log(torch.clamp(1.0-torch.abs(torch.erf((mu2 - mu1) / 2.0)), 1e-12, 1.0))
        ln_1_min_area = torch.log(torch.clamp(torch.abs(torch.erf((mu2 - mu1) / 2.0)), 1e-12, 1.0))
        #ln_1_min_area = torch.log(torch.clamp(prod, 1e-7, 1.0))

        '''
        fln_area = ln_gaussian_overlap.apply
        fln_1_min_area = ln_1min_gaussian_overlap.apply
        ln_area = fln_area(mu1-mu2)
        ln_1_min_area = fln_1_min_area(mu1 - mu2)
        '''

        ln_area = torch.mean(ln_area, 1).unsqueeze(1)
        ln_1_min_area = torch.mean(ln_1_min_area, 1).unsqueeze(1)
        loss = -y*ln_area - (1-y)*ln_1_min_area
        loss = torch.mean(loss)


        return loss





class KL_avg_sigma(torch.nn.Module):
    """
    Contrastive loss function.

    Based on:
    """

    def __init__(self):
        super(KL_avg_sigma, self).__init__()

    def forward(self, mu1, logsigma1, mu2, logsigma2, y):
        sigma1 = torch.exp(logsigma1)
        sigma2 = torch.exp(logsigma2)
        sigma = 0.5*(sigma1 + sigma2)

        ln_area = torch.log(torch.clamp(1.0-torch.abs(torch.erf((mu2 - mu1) / 2.0 / sigma)), 1e-12, 1.0))
        ln_1_min_area = torch.log(torch.clamp(torch.abs(torch.erf((mu2 - mu1) / 2.0 / sigma)), 1e-12, 1.0))


        loss = -y*torch.mean(ln_area, 1).unsqueeze(1)
        loss += -(1-y)*torch.mean(ln_1_min_area, 1).unsqueeze(1)
        loss = torch.mean(loss)

        #add KL loss from variational component
        #loss += 0.25 * torch.mean((sigma1 + sigma2 + torch.pow(mu1, 2) + torch.pow(mu2, 2) - 2. - 2*logsigma1 - 2*logsigma2))

        return loss


class KL_with_sigma(torch.nn.Module):
    """
    Contrastive loss function.

    Based on:
    """

    def __init__(self):
        super(KL_with_sigma, self).__init__()

    def forward(self, mu1in, logsigma1in, mu2in, logsigma2in, y):
        #logsigma1, logsigma2 = (mu1 < mu2).float()*logsigma1 + (mu1 >= mu2).float()*logsigma2, (mu1 >= mu2).float()*logsigma2 + (mu1 >= mu2).float()*logsigma1
        #mu1, mu2 = (mu1 < mu2).float()*mu1 + (mu1 >= mu2).float()*mu2, (mu1 < mu2).float()*mu2 + (mu1 >= mu2).float()*mu1

        #make curve 1 the narrower curve
        mu1 = (logsigma1in < logsigma2in).float()*mu1in + (logsigma1in >= logsigma2in).float()*mu2in
        mu2 = (logsigma1in < logsigma2in).float()*mu2in + (logsigma1in >= logsigma2in).float()*mu1in
        logsigma1 = (logsigma1in < logsigma2in).float()*logsigma1in + (logsigma1in >= logsigma2in).float()*logsigma2in
        logsigma2 = (logsigma1in < logsigma2in).float()*logsigma2in + (logsigma1in >= logsigma2in).float()*logsigma1in


        sigma1 = torch.exp(logsigma1)
        sigma2 = torch.exp(logsigma2)
        sigma1sq = torch.pow(sigma1, 2)
        sigma2sq = torch.pow(sigma2, 2)
        mu1sq = torch.pow(mu1, 2)
        mu2sq = torch.pow(mu2, 2)

        s = torch.pow(mu1*sigma2sq - mu2*sigma1sq, 2) + (sigma1sq - sigma2sq) * (mu1sq*sigma2sq - mu2sq*sigma1sq - 2*sigma1sq*sigma2sq*(logsigma1 - logsigma2))
        sqrtd = torch.sqrt(torch.clamp(s, 0.0, None))

        c_left = (mu2*sigma1sq - mu1*sigma2sq + sqrtd) / (sigma1sq - sigma2sq + 1.0e-6)
        c_right = (mu2 * sigma1sq - mu1 * sigma2sq - sqrtd) / (sigma1sq - sigma2sq + 1.0e-6)

        print(mu1[0, 0:4])
        print(mu2[0, 0:4])
        print(sigma1[0, 0:4])
        print(sigma2[0, 0:4])
        print(c_left[0, 0:4])
        print(c_right[0, 0:4])

        area = 0.5 * torch.erf((c_left - mu1) / np.sqrt(2.0) / sigma1)
        area -= 0.5 * torch.erf((c_right - mu1) / np.sqrt(2.0) / sigma1)
        area += 0.5 * torch.erf((c_right - mu2) / np.sqrt(2.0) / sigma2)
        area -= 0.5 * torch.erf((c_left - mu2) / np.sqrt(2.0) / sigma2)
        area += 1.0

        print(area[0, 0:4])
        print(area.max())
        print(torch.abs(mu1-mu2).min())

        area = torch.mean(area, 1).unsqueeze(1)

        loss = -y*torch.mean(torch.log(torch.clamp(area, 1.0e-7, 1.0)), 1).unsqueeze(1)
        loss += -(1-y)*torch.mean(torch.log(torch.clamp(1.0-area, 1.0e-7, 1.0)), 1).unsqueeze(1)
        loss = torch.mean(loss)
        loss += 0.5 * torch.mean((sigma1 + sigma2 + torch.pow(mu1, 2) + torch.pow(mu2, 2) - 2. - 2*logsigma1 - 2*logsigma2))

        return loss



class VarLoss(torch.nn.Module):
    def __init__(self):
        super(VarLoss, self).__init__()


    def forward(self, mu, logz):
        return torch.mean((1.0 + logz - z_mean*z_mean - torch.exp(logz)).sum(1)/2.0)
