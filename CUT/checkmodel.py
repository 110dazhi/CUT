import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util
import wandb
from models.networks import PatchSampleF
from torchsummary import summary

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    # train_dataset = create_dataset(util.copyconf(opt, phase="train"))
    model = create_model(opt)
    # for name, parameters in model.named_parameters():
    #   print(name, ':', parameters.size())
    # summary(model, (3, 256, 256))
    # help(model)
    net = model.print_networks(verbose=True)
    # net = model.net
    netG = model.netG
    summary(netG,(3,256,256),batch_size=1)
    netF = model.netD
    # dis = PatchSampleF(use_mlp=True, init_type="xavier", init_gain=0.02, gpu_ids=opt.gpu_ids, nc=256)
    print(netF)
    summary(netF,(256,256,3),batch_size=4)