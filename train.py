import numpy as np
import torch
import torch as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import socket
from torch.optim import optimizer
import argparse
import time
from torch.utils.tensorboard import SummaryWriter
from prefetch_generator import BackgroundGenerator
from genres import genrelist
from get_data import AudioDataset
from models import AudioGenerator
from criterions import melody_comparison, audio_quality_check

if __name__ == '__main__':
    # settings that can be changed with console command options
    parser = argparse.ArgumentParser(description='PyTorch ESRGANplus')
    parser.add_argument('--numDepthLayers', type=int, default=4, help=("number of Down/Upsampling layers (higher = better but computational less effiecient)"))
    parser.add_argument('--attentionMechanism', type=bool, default=False, help=("set if an Attention mechanism should be used or not"))
    parser.add_argument('--melodyCriterion', type=bool, default=True, help=("set if a Melody Criterion and Melody Extractor will be used"))
    parser.add_argument('--qualityCriterion', type=bool, default=True, help=("set if a Quality Criterion and Quality Check will be used"))
    parser.add_argument('--WithVoice', type=bool, default=False, help=("set if voiced music will be inputed or not (default = False)"))
    parser.add_argument('--seed', type=int, default=123, help=("set seed"))
    parser.add_argument('--gpu_mode', type=bool, default=False, help=("set cuda on/off"))
    parser.add_argument('--genreSpecificSaves', type=bool, default=True, help=("set if every genre gets its own AI weight saves"))
    parser.add_argument('--WAV_DIRECTORY_PATH', type=str, default="../../data/Music_data/train/", help=("set input file path"))
    parser.add_argument('--LABEL_DIRECTORY_PATH', type=str, default="../../data/Music_data/valid/", help=("set label file path"))
    parser.add_argument('--resume', type=bool, default=False, help=("resume training"))
    parser.add_argument('--nEpochs', type=int, default=10, help=("number of epochs"))
    parser.add_argument('--model_type', type=str, default="Music Gen", help=("Model Type"))
    parser.add_argument('--snapshots', type=int, default=10, help=("which epochs to save (default 10)"))
    parser.add_argument('--save_folder', type=str, default="../../results/Music_Generator", help=("model save folder"))
    parser.add_argument('--nBatches', type=int, default=4, help=("num of batches"))
    parser.add_argument('--lr', type=float, default=None, help=("learning rate"))
    parser.add_argument('--beta1', type=float, default=None, help=("beta 1"))
    parser.add_argument('--beta2', type=float, default=None, help=("beta 2"))
    parser.add_argument('--gpus', type=int, default=1, help=("Number of gpus"))
    parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch for continuing training')
    
    opt = parser.parse_args()
    np.random.seed(opt.seed)    # set seed to default 123 or opt
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    gpus_list = range(opt.gpus)
    hostname = str(socket.gethostname)
    cudnn.benchmark = True
    print(opt)  # print the chosen parameters

    if opt.genreSpecificSaves:
        genres = genrelist 
    else:
        genres = None   # if not using specific saves for each genre
        
    for genre in genres:    # get one weight save for each genre
    # data loading
        print('==> Loading Datasets')
        dataloader = DataLoader(AudioDataset(opt.WAV_DIRECTORY_PATH,opt.LABEL_DIRECTORY_PATH,genre))

        audio_generator = AudioGenerator()

        # print full model parameter count
        audio_generator_params = sum(p.numel() for p in audio_generator.parameters())
        print("Number of Audio generator Parameters: {}".format(audio_generator_params))
        
        # loss
        melody_criterion = melody_comparison()
        quality_criterion = audio_quality_check()

        # run on gpu
        cuda = opt.gpu_mode
        if cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")

        torch.manual_seed(opt.seed)
        if cuda:
            torch.cuda.manual_seed(opt.seed)

        if cuda:
            audio_generator = audio_generator.cuda(gpus_list[0])
            melody_criterion = melody_criterion.cuda(gpus_list[0])

        # optimizer
        optimizer_audio_generator = optim.Adam(audio_generator.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
        #optimizer_audio_generator = adafactor(audio_generator.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2)) try this for experiments
  
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
            torch.save(audio_generator.state_dict(), model_out_path)
            print("Checkpoint saved to {}".format(model_out_path))

        # multiple gpu run
        #if opt.multiGPU:
        #    audio_generator = torch.nn.DataParallel(audio_generator, device_ids=gpus_list)

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

                # prepare data
                wave_x = audio["wave_x"]
                sample_rate_x = audio["sample_rate_x"]
                wave_y = audio["wave_y"]
                sample_rate_y = audio["sample_rate_y"]

                if cuda:    # put variables to gpu
                    wave_x = wave_x.to(gpus_list[0])
                    sample_rate_x = sample_rate_x.to(gpus_list[0])
                    wave_y = wave_y.to(gpus_list[0])
                    sample_rate_y = sample_rate_y.to(gpus_list[0])
                    
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
            Net = audio_generator

        print('----------------Network architecture----------------')
        print_network(Net)
        print('----------------------------------------------------')