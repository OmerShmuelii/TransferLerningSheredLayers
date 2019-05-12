import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from MobileNetV2 import MobileNetV2
from tensorboardX import SummaryWriter
import numpy as np
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

model_names.append('mobilenetV2')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data',nargs='?',default="Dataset/train/Cropped", metavar='DIR',     type=str, #Cropped
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='mobilenetV2',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: mobilenetV2)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--epochs', default=850, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=70, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,    # 1e-5
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,            #1e-4
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--lrd','--learning-rate-decay',  default=0.98, type=float,            #1e-4
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

best_prec1 = 0

writer = SummaryWriter()
learnOnlyFirst = True  # Learn only first network banchmark
therlist=[38, 70, 151, 281, 300, 327, 389, 398, 488, 578, 668, 758, 845, 936, 958, 1000]  # section to split the data
Lamdada=[20, 1,  1,    1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1]    #weights for the weighted mean everage
CheckPointName='checkpoint13b.pth.tar'
useSaveCheckPoint=True
LoadModelFromPreTrainedMobilenetv2=False
PreTrainedMobilenetv2='mobilenet_v2.pth.tar'
useparamoptimized=False # freeze most of the model
    #np.arange(62.5,1010,62.5)

def adjust_learning_rates( arg, optimizer, epoch):
    """Sets the learning rate to the initial LR multiplied by 0.98 every epoch"""
    learning_rate = arg.lr * (arg.lrd ** epoch)
    writer.add_scalar('data/LearningRate', learning_rate, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

def loadNetUseFirst(net):
    allModel = torch.load(PreTrainedMobilenetv2)  # add map_location='cpu' if no gpu
    # state_dict = torch.save("mobilenetMy_v2.pth.tar")
    # metadata = getattr(state_dict, '_metadata', None)
    state_dict = allModel
    if 0:
        state_dict= allModel['state_dict']
        optimizers=    allModel['optimizer']
        entireModel=allModel['EntireModel']
        optimizer_state_dict=optimizers['state']
    #optimizer = torch.optim.RMSprop(optimizer_state_dict)
   # optimizer.load_state_dict(optimizers)#entireModel.parameters())
    model_dict = net.state_dict()
    if 0:
        enti=net.state_dict().keys()#{name.rsplit('.', 1)[0] for  name in entireModel.state_dict() }
        StateDictDump={}
        for name, v in model_dict.items():
            if not('module.'+name in state_dict and v.size()==state_dict['module.'+name].size()):
                StateDictDump['module.'+name] =v
            else:
                if name.split('.', 1)[0] == 'classifier':
                    print (name)
                    print(state_dict['module.' + name].size())
        StateDictDump = {name:v for  name,v in model_dict.items() if not
                           (name  in state_dict )}
    if 1:
        last = '0'
        childs = []
        index = 0

        for name, child in net._modules.items():
            # prefix.append(name)
            if name == "features2":
                break
            if name == "SplitLayer":
                continue
            if child is not None:
                for namei, childi in child._modules.items():
                    last = int(namei)

        keys = []
        con = []
        state_dict2 = {}
        for key in state_dict:
            keys.append(key)
            con.append(state_dict[key])
            listi = key.split('.')
            key2 = ''
            if (listi[0] == "features") & (int(float(listi[1])) > last):
                for part in range(0, len(listi)):
                    if part == 0:
                        listi[part] += '2'
                    if part == 1:
                        listi[part] = str(int(float(listi[1])) - last - 1)
                    key2 += listi[part]
                    if part < len(listi) - 1:
                        key2 += '.'

            else:
                key2 = key
            state_dict2[key2] = state_dict[key]


        state_dict2Old=state_dict2
    else:
        state_dict2Old= state_dict
    # 1. filter out unnecessary keys
    initialMap=True
    if    initialMap:
        state_dict2 = {k: v for k, v in state_dict2Old.items() if    #.split('.',1)[1]
                       (k in model_dict and model_dict[k].size() == v.size())} #k.split('.',1)[1] k.split('.',1)[1]#
        state_dict2Out = {k: v for k, v in model_dict.items() if len(v.size()) >0 and not (k in state_dict2Old and (state_dict2Old[k].size() == v.size()))}          # or
        for k in state_dict2Out:
            print (len(model_dict[k].size()))
            print(k)
    else:
        state_dict2 = {k.split('.',1)[1]: v for k, v in state_dict2Old.items() if  # .
                       (k.split('.',1)[1] in model_dict and model_dict[k.split('.',1)[1]].size() == v.size())}  # k.split('.',1)[1] k.split('.',1)[1]#
        state_dict2Out = {k for k, v in model_dict.items() if not
                       ('module.'+k in state_dict2Old and state_dict2Old['module.'+k].size() == v.size())}
        # 1. filter out unnecessary keys
   # state_dictOut2 = {k for k, v in state_dict2.items()}  # if
   # state_modelOut2 = {k for k, v in model_dict.items()}# if
                     #   (k not in state_dict2.items())}
    # 2. overwrite entries in the existing state dict
    model_dict.update(state_dict2)
    # 3. load the new state dict
    net.load_state_dict(model_dict)
    return state_dict2Out

def main():
    global args, best_prec1
    args = parser.parse_args()

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        if args.arch.startswith('mobilenetV2'):
            model = MobileNetV2(n_class=1001)
            if LoadModelFromPreTrainedMobilenetv2:
                state_dictOut2=loadNetUseFirst(model)
                paramsOptimized={p for name,p in model.named_parameters() if
                 (name in state_dictOut2)}
                if 0:
                    Best_Model = torch.load("checkpoint.pth.tar")
                    state_dict=  Best_Model["state_dict"]
                    model_dict = model.state_dict()
                    state_dictInModel = {k: v for k, v in state_dict.items() if
                                   (k in model_dict)}
                    model_dict.update(state_dictInModel)
                    model.load_state_dict(model_dict)
                    args.start_epoch = Best_Model["epoch"]
                # new_params = []
                # new_paramsNames = []
                # for name,p in model.named_parameters():
                #     new_params.append(p)#model.parameters()
                #     new_paramsNames.append((name))
            print(model)

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    #optimizer = torch.optim.Adam(model.parameters(),args.lr,(0.9, 0.999),0.1,0.9)
   # optimizer = torch.optim.SGD(model.parameters(), args.lr, 0.9)# (0.9, 0.999), 0.1, 0.9)#paramsOptimized

    args.resume =CheckPointName #'checkpoint5.pth.tar'
    # optionally resume from a checkpoint
    if useSaveCheckPoint:#args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            optimizer = torch.optim.RMSprop( model.parameters(), args.lr, momentum=args.momentum,
                                            weight_decay=args.weight_decay)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        optimizer = torch.optim.RMSprop( model.parameters(), args.lr, momentum=args.momentum,
                                        weight_decay=args.weight_decay)  # model.parameters()   paramsOptimized
    if useparamoptimized:
        optimizer = torch.optim.RMSprop(paramsOptimized, args.lr, momentum=args.momentum,
                                        weight_decay=args.weight_decay)

    cudnn.benchmark = True
   # blblbl=model.features2.state_dict()#keep_vars=True)
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
           # transforms.Scale(256),
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=int(3.5*args.batch_size), shuffle=False,  #10
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return
    else:
        if 0:
            validate(val_loader, model, criterion)
    #args.lr=  args.lr
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rates(args, optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1, precRel = validate(val_loader, model, criterion)
        writer.add_scalar('data/prec1', prec1.item(), epoch)
        writer.add_scalar('data/precRel', precRel.item(), epoch)
        # remember best prec@1 and save checkpoint
        is_best = precRel > best_prec1
        best_prec1 = max(prec1, best_prec1)
        #adjust_learning_rates(args, optimizer, epoch)
        # args.lr=args.lr-
        # optimizer = torch.optim.RMSprop(paramsOptimized, args.lr,  # model.parameters()
        #                                 momentum=args.momentum,
        #                                 weight_decay=args.weight_decay)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'EntireModel': model,
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best)

    writer.close()
def splitData(input,target):
    cur_batch_size=input.size()[0]
    chunkAll = torch.chunk(input, cur_batch_size)
    isfirst=[]
    for thr in therlist:
        isfirst.append(True)
    chunkFainelalllist = [None] * 16
    targetorgrel= [None] * 16
    targetorgrelFixed = [None] * 16
    for k in range(0, cur_batch_size):
        throld = int(0)
        for i, thr in enumerate(therlist):
            if target[k] < thr:
                if not isfirst[i]:
                    chunkFainelalllist[i] = torch.cat((chunkFainelalllist[i], chunkAll[k]), 0)
                    targetorgrel[i] += [target[k].item()]
                    targetorgrelFixed[i] += [target[k].item()-throld]
                else:
                    chunkFainelalllist[i] = chunkAll[k]
                    targetorgrel[i] = [target[k].item()]
                    targetorgrelFixed[i] = [target[k].item() - throld]
                    isfirst[i] = False
                break
            throld = int(np.ceil(thr))
        # else:
        #     if not isfirst2:
        #         chunkFainelalllist[0] = torch.cat((chunkFainelalllist[0], chunkAll[k]), 0)
        #         targetorgrel += [target[k].item()]  # torch.cat(, 0)
        #     else:
        #         chunkFainelalllist[0] = chunkAll[k]
        #         targetorgrel = [target[k].item()]
        #         isfirst2 = False
    sizevec=[]
    targetorgrelremerge=[]
    targetorgrelremergeFix = []
    for  i, target in enumerate(targetorgrel):
        if target !=   None:
            sizevec.append(len(target))
            targetorgrelremerge+=target
            targetorgrelremergeFix+= targetorgrelFixed[i]
            if not 'chunkFainelallmerge' in vars():
                chunkFainelallmerge= chunkFainelalllist[i]
            else:
                chunkFainelallmerge=torch.cat((chunkFainelallmerge,chunkFainelalllist[i]),0)
        else:
            sizevec.append(0)
    # if not 'targetorgrel' in vars():
    #     size1=0
    # else:
    #     size1 = len(targetorgrel[0])
    # if not 'targetorgrel2' in vars():
    #     size2 = 0
    # else:
    #     size2 = len(targetorgrel[1])
    # if not 'targetorgrel' in vars():
    #     if 'targetorgrel2' in vars():
    #         targetorgrel[0] = targetorgrel[1]
    # else:
    #     if 'targetorgrel2' in vars():
    #         targetorgrel[0] += targetorgrel[1]
    # if 'chunkFainel' in vars():
    #     if 'chunkFainel2' in vars():
    #         return size1,size2,targetorgrel, torch.cat((chunkFainelalllist[0], chunkFainelalllist[1]), 0)
    #     else:
    #         return size1, size2, targetorgrel, chunkFainelalllist[0]
    # else:
    return sizevec, targetorgrelremerge, targetorgrelremergeFix, chunkFainelallmerge

def remerge(outputfirst,sizevec,learnOnlyFirstinput):
    chunk = torch.chunk(outputfirst, 16, 1)
    if learnOnlyFirstinput:
        return chunk[0]
    sizetotal=0
    for i, size in enumerate(sizevec):
        if size>0:
            current = chunk[i][sizetotal:sizetotal+size, :, :, :]
            sizetotal += size
            if not 'Total' in vars():
                Total=current
            else:
                Total = torch.cat((Total, current), 0)
    # if size1 > 0:
    #     first = chunk[0][:size1, :, :, :]
    # if size2 > 0:
    #     sec = chunk[1][size1:size1 + size2, :, :, :]
    #     if size1 > 0:
    #         return torch.cat((first, sec), 0)
    #     else:
    #         return sec
    # else:
    #     if size1 > 0:
    #         return first
    return Total#chunk[0]


def remergeSplit(Last,sizevec):
    ceilterlist=[]
    last=0

    for val in np.ceil(therlist).astype(int):
        ceilterlist.append(val-last)
        last= val
        vals = Last.view(-1)
        minarg = torch.argmin(vals)
        minval=  vals[minarg]
    #ceilterlist=tuple((np.floor(therlist[:-1]).astype(int)))
    chunk = torch.split(Last[:,:-1], ceilterlist, 1)
    sizetotal=0
    Total = [None] * 16
    for i, size in enumerate(sizevec):
        if size>0:

           # current=torch.empty(size, 63, dtype=torch.float).fill_(minval.item())
           # current[:,:chunk[i].size(1)] = chunk[i][sizetotal:sizetotal+size]
            current = chunk[i][sizetotal:sizetotal + size]
            sizetotal += size
            Total[i] = current
            #if not 'Total' in vars():
            #    Total=current
            #else:
            #   Total = torch.cat((Total, current), 0)
    return Total#chunk[0]
def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    criterionRes=  nn.MSELoss()
    TrainRestore=False
    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)



        #arget_var = torch.autograd.Variable(target.repeat([16]))

        # compute output

        #outputlocal=  outputfirst

        totalloss=0
        count=0
        optimizer.zero_grad()
        sizevec, targetorgrel, targetorgrelFixed, input=splitData(input, target)
        #targetorgrel=[]
       # targetorgrel2 = []
        if sizevec[0]<1:
            continue


        input_var = torch.autograd.Variable(input)


            #torch.cat((torch.chunk(chunkFainel, 16, 1)[0],torch.chunk(chunkFainel2, 16, 1)[1]),0)
        #targeti = torch.from_numpy(np.asarray(targetorgrel)).long().cuda(async=True)
        targeti = torch.from_numpy(np.asarray(targetorgrelFixed)).long().cuda(async=True)
        targetorgrel=None
        target_var1 = torch.autograd.Variable(targeti)
        outputfirst = model(input_var, 3)
        if TrainRestore:
            optimizer.zero_grad()
            chunks = torch.chunk(outputfirst, 16, 1)
            Restored=None
            for kok in range(16):
                if Restored is None:
                    Restored=model(chunks[kok], 5)
                else:
                    Restored =Restored+ model(chunks[kok], 5)
            #lossRes=Restored.cuda() - input_var.cuda()
            #lossRes =  (lossRes**2).mean()#
            lossRes = criterionRes(Restored, input_var.cuda())
            totalloss = lossRes.item()
            lossRes.backward()
            optimizer.step()
            writer.add_scalar('data/trainLoss' + str(epoch), totalloss, i)
            if i%10==0:
                print (str(epoch)+ ' ['+str(i)+', '+ str(len(train_loader))+ '] loss:'+str(totalloss) )
            continue

        bouth=remerge(outputfirst, sizevec,learnOnlyFirst)
        learn_Only_Base = True
        if learn_Only_Base:#for chunk in  uotputchunk:
            output = model(bouth, 4)
            #check=output[0,0:64]
            outputReArrange=remergeSplit(output,sizevec)
           # losscollect.append(
           # targeti = torch.from_numpy(np.asarray(targetorgrel)).long().cuda(async=True)
            loss=0
            totalsize=0
            prec1=0
            prec5=0
            precRel=0
            sizTtotal = 0
            lambadaNum=0
            for size in sizevec:
                sizTtotal += size * Lamdada[lambadaNum]
                lambadaNum=lambadaNum+1

            for categind in range(16) :
                if sizevec[categind]>0:
                    loss+=criterion(outputReArrange[categind].cuda(), target_var1[totalsize:totalsize+sizevec[categind]])*sizevec[categind]*Lamdada[categind] /sizTtotal
                    preccur1, preccur5 = accuracy(outputReArrange[categind].data, targeti[totalsize:totalsize+sizevec[categind]], topk=(1, 5))
                    prec5+= preccur5*sizevec[categind]/sizTtotal
                    prec1 += preccur1*sizevec[categind]/sizTtotal
                    if categind==0:
                        writer.add_scalar('Pred/trainPred' + str(epoch), preccur1, i)
                    totalsize += sizevec[categind]
            #)
            totalloss =loss.item()# losscollect[count].data[0]

            # measure accuracy and record loss

            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))
            #writer.export_scalars_to_json("all_scalars.json")
            #  Optimization step
            count = count + 1

            if count<16 and not learn_Only_Base:
                loss.backward(retain_graph=True)
            else:
                loss.backward()


        # loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        output = None
        outputfirst=None
        uotputchunk=None
        chunkFainel = None
        chunkFainel2 = None
        chunkAll=None
        chunk=None
        # for lospart in losscollect:
        #
        #     if  count2<15 :
        #         lospart.backward(retain_graph=True)       #
        #     else:
        #         lospart.backward(retain_graph=False)
        #
        #     count2=count2+1

        statedict=model.state_dict()
        totalloss=totalloss #/ 16

        losses.update(totalloss, input.size(0))
        writer.add_scalar('data/trainLoss' + str(epoch), totalloss, i)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    top1Rel = AverageMeter()
    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        sizevec, targetorgrel, targetorgrelFixed, input = splitData(input, target)

        target = torch.from_numpy(np.asarray(targetorgrelFixed)).long().cuda(async=True)# target.cuda(async=True)#.repeat([16])
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        with torch.no_grad():
            # compute output
            outputFirst = model(input_var, 3)
            bouth = remerge(outputFirst, sizevec,learnOnlyFirst)
            output = model(bouth,4)
            outputReArrange = remergeSplit(output, sizevec)#.cuda()
            #loss = criterion(outputReArrange, target_var)
            loss = 0
            totalsize = 0
            prec1 = 0
            prec5 = 0
            sizTtotal=0
            for size in sizevec:
                sizTtotal+= size
            for categind in range(16):
                if sizevec[categind] > 0:
                    loss += criterion(outputReArrange[categind].cuda(),
                                      target_var[totalsize:totalsize + sizevec[categind]])*sizevec[categind] /sizTtotal
                    preccur1, preccur5 = accuracy(outputReArrange[categind].data,
                                                  target[totalsize:totalsize + sizevec[categind]], topk=(1, 5))
                    prec5 += preccur5 * sizevec[categind] / sizTtotal
                    prec1 += preccur1 * sizevec[categind] / sizTtotal
                    if categind==0:
                        writer.add_scalar('Pred/ValidPred' , preccur1, i)
                        top1Rel.update(preccur1[0], sizevec[categind])
                    totalsize += sizevec[categind]
            # )
        writer.add_scalar('data/ValidLoss', loss.item(), i)
        # measure accuracy and record loss
       # prec1, prec5 = accuracy(outputReArrange.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        print('(min,max,prec1,prec5)= ', min(targetorgrel), max(targetorgrel), prec1[0], prec5[0])
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg ,top1Rel.avg


def save_checkpoint(state, is_best, filename='checkpoint13b.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best13b.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()