import numpy as np
import torch
import torch as nn
from torch.nn.modules.loss import L1Loss
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import socket
import argparse
import time
from torch.utils.tensorboard import SummaryWriter
from prefetch_generator import BackgroundGenerator
from genres import genrelist
from get_data import AudioDataset


if __name__ == '__main__':
    # settings that can be changed with console command options
    parser = argparse.ArgumentParser(description='PyTorch ESRGANplus')
    #parser.add_argument('--ray_tune', type=bool, default=False, help=("Use ray tune to tune parameters"))
   
    opt = parser.parse_args()
    np.random.seed(opt.seed)    # set seed to default 123 or opt
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    gpus_list = range(opt.gpus)
    hostname = str(socket.gethostname)
    cudnn.benchmark = True
    print(opt)  # print the chosen parameters

    for genre in genrelist: # get one weight save for each genre
        
    # data loading
        print('==> Loading Datasets')
        dataloader = DataLoader(AudioDataset(genre))

        pytorch_single_track_generator_params = sum(p.numel() for p in SingleTrackGenerator.parameters())
        print("Number of Instrument Choose Params: {},Number of Track Merge Params: {}, Number of singler Gen Params:{}".format(pytorch_single_track_generator_params))
        # loss

        # run on gpu
        cuda = opt.gpu_mode
        if cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")

        torch.manual_seed(opt.seed)
        if cuda:
            torch.cuda.manual_seed(opt.seed)

        if cuda:
            SingleTrackGenerator = SingleTrackGenerator.cuda(gpus_list[0])
            content_criterion = content_criterion.cuda(gpus_list[0])

        # optimizer
        optimizerSingleTrack = optim.Adam(SingleTrackGenerator.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))  # ESRGANplus / generator optimizer
    
        # load checkpoint/load model
        star_n_iter = 0
        start_epoch = 0
        if opt.resume:
            checkpoint = torch.load(opt.save_folder) ## look at what to load
            start_epoch = checkpoint['epoch']
            start_n_iter = checkpoint['n_iter']
            optimizer.load_state_dict(checkpoint['optim'])
            print("last checkpoint restored")

        def checkpoint(epoch):
            model_out_path = opt.save_folder+opt.model_type+genre+".pth".format(epoch) # google multi model saving
            torch.save(SingleTrackGenerator.state_dict(), model_out_path)
            print("Checkpoint saved to {}".format(model_out_path))

        # multiple gpu run
        if opt.multiGPU:
            SingleTrackGenerator = torch.nn.DataParallel(SingleTrackGenerator, device_ids=gpus_list)

        # tensor board
        writer = SummaryWriter()

        # define Tensor
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        for epoch in range(start_epoch, opt.nEpochs):
            epoch_loss = 0
            start_time = time.time()

            # prefetch generator and tqdm for iterating trough data
            


            for i, audio in enumerate(BackgroundGenerator(dataloader,1)):   #  for data in pbar # Count, item in enumerate
                
                # data preparation
                #pbar(i, len(dataloader))
                # prepare data
                waveform = audio["waveform"]
                sample_rate = audio["sample_rate"]

                if cuda:    # put variables to gpu
                    imgs_lr = imgs_lr.to(gpus_list[0])

                # keep track of prepare time
                prepare_time = time.time() - start_time

                #train generator  

                # update tensorboard


                #compute time and compute efficiency and print information
                process_time = time.time() - start_time
                print("process time: {}, Number of Iteration {}/{}".format(process_time,i , (len(dataloader)-1)))
                #pbar.set_description("Compute efficiency. {:.2f}, epoch: {}/{}".format(process_time/(process_time+prepare_time),epoch, opt.epoch))
                start_time = time.time()

            if (epoch+1) % (opt.snapshots) == 0:
                checkpoint(epoch)
            writer.add_scalar('loss', Loss)
            print("===> Epoch {} Complete: Avg. loss: {:.4f}".format(epoch, ((epoch_loss/2) / len(dataloader))))

        def print_network(net):
            num_params = 0
            for param in net.parameters():
                num_params += param.numel()
            print(net)
            print('Total number of parameters: %d' % num_params)

        print('===> Building Model ', opt.model_type)
        if opt.model_type == 'ESRGANplus':
            Net = SingleTrackGenerator

        print('----------------Network architecture----------------')
        print_network(Net)
        print('----------------------------------------------------')