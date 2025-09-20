import os
import torch.utils.data
from torch import nn
# from torch.nn import DataParallel
from datetime import datetime
from torch.optim import lr_scheduler
import torch.optim as optim
import time
# import numpy as np
import argparse
import sys
from pathlib import Path

from dataloader.testloader import LFW
# import tensorboard

from model_architecture import model
from model_architecture.utils import init_log
from dataloader.loaddata import DataLoader
import scipy
import numpy as np

from lfw_eval import parseList, evaluation_10_fold


# from config import BATCH_SIZE, SAVE_FREQ, RESUME, SAVE_DIR, TEST_FREQ, TOTAL_EPOCH, MODEL_PRE, GPU
# from config import CASIA_DATA_DIR, LFW_DATA_DIR

def define_gpu(gpu):
    # gpu init
    gpu_list = ''
    multi_gpus = False
    if isinstance(gpu, int):
        gpu_list = str(gpu)
    else:
        multi_gpus = True
        for i, gpu_id in enumerate(gpu):
            gpu_list += str(gpu_id)
            if i != len(gpu) - 1:
                gpu_list += ','
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

    return multi_gpus


def training(model_pre, save_dir, batch_size, total_epoch, resume, gpu, data_dir):
    multi_gpus = define_gpu(gpu)
    print('multi_gpus', multi_gpus)

    # other init
    start_epoch = 1
    save_dir = os.path.join(save_dir, model_pre + datetime.now().strftime('%Y%m%d_%H%M%S'))
    if os.path.exists(save_dir):
        raise NameError('model dir exists!')
    os.makedirs(save_dir)
    logging = init_log(save_dir)
    _print = logging.info

    # define trainloader and testloader
    print('defining casia dataloader...')
    trainset = DataLoader(root=data_dir)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=8, drop_last=False)

    # nl: left_image_path
    # nr: right_image_path
    # print('defining lfw dataloader...')
    # nl, nr, folds, flags = parseList(root="C:/Users/daoda/OneDrive/Documents/TestImage")
    # testdataset = LFW(nl, nr)
    # testloader = torch.utils.data.DataLoader(testdataset, batch_size=32,
    #                                          shuffle=False, num_workers=8, drop_last=False)

    # define model
    print('defining shufflefacenet model...')
    net = model.ShuffleFaceNet()

    # if resume:
    #     ckpt = torch.load(resume, map_location=torch.device('cpu'))
    #     net.load_state_dict(ckpt['net_state_dict'])
    #     start_epoch = ckpt['epoch'] + 1

    # net = net.cuda()
    # net = net
    # # NLLLoss
    nllloss = nn.CrossEntropyLoss()
    # # nllloss = nn.CrossEntropyLoss().cuda()
    # # CenterLoss
    lmcl_loss = model.CosFace_loss(num_classes=trainset.class_nums, feat_dim=128)
    # # lmcl_loss = model.CosFace_loss(num_classes=trainset.class_nums, feat_dim=128).cuda()
    #
    # if multi_gpus:
    #     net = DataParallel(net)
    #     lmcl_loss = DataParallel(lmcl_loss)
    #
    criterion = [nllloss, lmcl_loss]

    checkpoint = torch.torch.load("model/best/060.ckpt", map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['net_state_dict'])
    net.eval()
    i = 213
    for param in net.parameters():
        if i > 3:
            param.requires_grad = False
            i -= 1

    # optimzer4nn
    optimizer4nn = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0005)
    sheduler_4nn = lr_scheduler.StepLR(optimizer4nn, 20, gamma=0.5)

    # optimzer4center
    optimzer4center = optim.Adam(lmcl_loss.parameters(), lr=0.01)
    sheduler_4center = lr_scheduler.StepLR(optimizer4nn, 20, gamma=0.5)

    best_acc = 0.0
    best_epoch = 0
    for epoch in range(start_epoch, total_epoch + 1):
        # exp_lr_scheduler.step()
        optimizer4nn.step()
        optimzer4center.step()
        # train model
        _print('Train Epoch: {}/{} ...'.format(epoch, total_epoch))
        net.train()

        train_total_loss = 0.0
        total = 0
        since = time.time()
        for data in trainloader:
            img, label = data[0], data[1]
            # img, label = data[0].cuda(), data[1].cuda()
            batch_size = img.size(0)
            # optimizer_ft.zero_grad()

            raw_logits = net(img)

            logits, mlogits = criterion[1](raw_logits, label)
            total_loss = criterion[0](mlogits, label)

            optimizer4nn.zero_grad()
            optimzer4center.zero_grad()

            total_loss.backward()

            optimizer4nn.step()
            optimzer4center.step()

            train_total_loss += total_loss.item() * batch_size
            total += batch_size

        train_total_loss = train_total_loss / total
        time_elapsed = time.time() - since
        loss_msg = '    total_loss: {:.4f} time: {:.0f}m {:.0f}s' \
            .format(train_total_loss, time_elapsed // 60, time_elapsed % 60)
        _print(loss_msg)

        # test model on lfw
        # if epoch % 10 == 0:
        #     net.eval()
        #     featureLs = None
        #     featureRs = None
        #     _print('Test Epoch: {} ...'.format(epoch))
        #     for data in testloader:
        #         for i in range(len(data)):
        #             data[i] = data[i]
        #         res = [net(d).data.cpu().numpy() for d in data]
        #         featureL = np.concatenate((res[0], res[1]), 1)
        #         featureR = np.concatenate((res[2], res[3]), 1)
        #         if featureLs is None:
        #             featureLs = featureL
        #         else:
        #             featureLs = np.concatenate((featureLs, featureL), 0)
        #         if featureRs is None:
        #             featureRs = featureR
        #         else:
        #             featureRs = np.concatenate((featureRs, featureR), 0)
        #
        #     result = {'fl': featureLs, 'fr': featureRs, 'fold': folds, 'flag': flags}
        #
        #     # save tmp_result
        #     # scipy.io.savemat('result/tmp_result.mat', result)
        #     # accs = evaluation_10_fold('result/tmp_result.mat')
        #     # _print('    ave: {:.4f}'.format(np.mean(accs) * 100))
        #     print(result)

        # save model
        if epoch % 10 == 0:
            msg = 'Saving checkpoint: {}'.format(epoch)
            _print(msg)
            if multi_gpus:
                net_state_dict = net.module.state_dict()
            else:
                net_state_dict = net.state_dict()
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            torch.save({
                'epoch': epoch,
                'net_state_dict': net_state_dict},
                os.path.join(save_dir, '%03d.ckpt' % epoch))
    print('finishing training')


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_pre', nargs='+', type=str, default='100.ckpt',
                        help='best model.ckpt path(s)')
    parser.add_argument('--data_dir', type=str, default='C:/Users/daoda/OneDrive/Documents/Anh do an',
                        help='data source')  # file/folder, 0 for webcam

    parser.add_argument('--batch_size', default=25, help='batch size')
    parser.add_argument('--epoch', default=60, help='trainning epoch')
    parser.add_argument('--resume', default='', help='continue tranning')
    parser.add_argument('--gpu', default=0, help='number of gpu')
    parser.add_argument('--save_dir', default='./model', help='save results')
    opt = parser.parse_args()

    training(opt.model_pre, opt.save_dir, opt.batch_size, opt.epoch, opt.resume, opt.gpu, opt.data_dir)
