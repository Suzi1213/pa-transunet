import argparse
import copy
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import BCELoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss, fast_hist
from utils import test_single_volume
from torchvision import transforms

def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    # 日志记录
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    # 给定学习率、分类、和batch_size
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    # 训练用的数据集
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))
    
    # db_test = Synapse_dataset(base_dir=args.volume_path, list_dir=args.list_dir, split="test_vol",
    #                            transform=transforms.Compose(
    #                                [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    db_test = Synapse_dataset(base_dir=args.volume_path, list_dir=args.list_dir, split="test_vol")
    print("The length of test set is: {}".format(len(db_test)))

    # 这个函数是用来初始化随机函数的，worker_id 是一个整数，范围在[0-num_class-1]
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    # 为了防止最后一个batch不够的情况，
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    testloader = DataLoader(db_test, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    # best_model_wts = copy.deepcopy(model.state_dict())
    if args.n_gpu > 1:
        # 使用多gpu来进行训练
        model = nn.DataParallel(model)
    

    # 经典交叉熵损失函数，此函数在面对样本不平衡的时候很nice
    bce_loss = BCELoss()
    # dice函数损失
    dice_loss = DiceLoss(num_classes)
    # 随机梯度下降
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    hist = np.zeros((args.num_classes, args.num_classes))
    iter_num_test = 0
    # max_epoch = args.max_epochs
    # max_iterations = args.max_epochs * len(testloader)  # max_epoch = max_iterations // len(trainloader) + 1
    # logging.info("{} iterations per epoch. {} max iterations ".format(len(testloader), max_iterations))
    # iterator = tqdm(range(max_epoch), ncols=70)
    # 经典交叉熵损失函数，此函数在面对样本不平衡的时候很nice 此函数主要针对多分类，二分类换成BCE loss
    bce_loss_test = BCELoss()
    # dice函数损失
    dice_loss_test = DiceLoss(num_classes)
    hist_test = np.zeros((args.num_classes, args.num_classes))
    best_test_acc = 0
    for epoch_num in iterator:
        model.train()
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            # label_batch.cuda()加了cuda（）的会把Tensor放在gpu的内存中
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = bce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            predictions = outputs.data.max(1)[1].squeeze(1).squeeze(0).cpu().numpy()
            gts = label_batch.data[:].squeeze(0).cpu().numpy()
            hist += fast_hist(label_pred=predictions.flatten(), label_true=gts.flatten(),
                              num_classes=args.num_classes)
            train_acc = np.diag(hist).sum() / hist.sum()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f, train_acc : %f' % (iter_num, loss.item(), loss_ce.item(), train_acc))


        model.eval()
        test_acc_list = []
        # for epoch_num in iterator:
        for i_batch_test, sampled_batch_test in enumerate(testloader):
            image_batch_test, label_batch_test = sampled_batch_test['image'], sampled_batch_test['label']
            # label_batch.cuda()加了cuda（）的会把Tensor放在gpu的内存中
            image_batch_test, label_batch_test = image_batch_test.cuda(), label_batch_test.cuda()
            with torch.no_grad():
                outputs_test = model(image_batch_test)
            loss_ce_test = bce_loss_test(outputs_test, label_batch_test[:].long())
            loss_dice_test = dice_loss_test(outputs_test, label_batch_test, softmax=True)
            loss_test = 0.5 * loss_ce_test + 0.5 * loss_dice_test
            predictions_test = outputs_test.data.max(1)[1].squeeze(1).squeeze(0).cpu().numpy()
            gts_test = label_batch_test.data[:].squeeze(0).cpu().numpy()
            hist_test += fast_hist(label_pred=predictions_test.flatten(), label_true=gts_test.flatten(),
                              num_classes=args.num_classes)
            test_acc = np.diag(hist_test).sum() / hist_test.sum()
            test_acc_list.append(test_acc)
            iter_num_test = iter_num_test + 1
            # logging.info('iteration %d : test_acc : %f' % (iter_num1, test_acc))
            logging.info('iteration %d : loss : %f, loss_ce: %f, test_acc : %f' % (iter_num_test, loss_test.item(), loss_ce_test.item(), test_acc))
        av_test_acc = np.nanmean(test_acc_list)
        logging.info("av_test_acc********:".format(av_test_acc))

            # test_acc += test_acc
            # print('test_acc1111111:' + str(test_acc))
            # test_acc = test_acc / len(testloader)
            # print('test_acc2222222:' + str(test_acc))
        # logging.info('测试平均acc : test_acc : %f' % (test_acc))
        
        if av_test_acc > best_test_acc:
            best_test_acc = av_test_acc
            logging.info("best_test_acc=====:".format(best_test_acc))
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path) 
            
        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break


        
    writer.close()
    return "Training Finished!"