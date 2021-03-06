import argparse
from train import train
#from evaluate import evaluate
from validate import validate

def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e","--epochs",type=int,dest="epochs",help="number of epochs",default=100)
    parser.add_argument("-lr",type=float,dest="lr",help="learning rate",default=0.001)
    #parser.add_argument("-d","--decay",type=float,dest="decay",help="weight decay",default=0.005)
    #parser.add_argument("-m","--momentum",type=float,dest="momentum",help="learning momentum",default=0.9)
    parser.add_argument("-n","--n_class", type=int, dest="n_class", help="number of segments", default=1)
    parser.add_argument("-i","--in_channel", type=int, dest="in_channel", help="number of input channels", default=1)
    parser.add_argument("--display", action = 'store_true')
    parser.add_argument("--save", action = 'store_true')
    parser.add_argument("--save-file", dest="save_file", default=None)
    parser.add_argument("--load", action = 'store_true')
    parser.add_argument("--load-file", dest="load_file", default=None)
    parser.add_argument("--eval", action = 'store_true')
    parser.add_argument("--validate", action = 'store_true')
    parser.add_argument("--directory", dest="directory", help="directory where training data is stored", default='../Data/train/')
    parser.add_argument("--loss", dest="loss", help="choose loss function", default='BCE')
    parser.add_argument("--image-size", type=int, dest="img_size", help="choose input image size", default=None)
    parser.add_argument("--dataset-size", type=int, dest="data_size", help="number of images in an epoch", default=None)
    args = parser.parse_args()
    if args.validate:
        validate(args.lr, args.n_class, args.in_channel, args.loss, args.display, directory=args.directory, img_size=args.img_size, data_size=args.data_size, load_file=args.load_file)
    else:
        train(args.epochs, args.lr, args.n_class, args.in_channel, args.loss, args.display, save=args.save, load=args.load, directory=args.directory, img_size=args.img_size, data_size=args.data_size, load_file=args.load_file, save_file=args.save_file)
    #if args.eval:
    #    evaluate()
    #elif args.validate:
    #    validate(args.display)
    #else:
    #    train(args.epochs, args.lr, args.n_class, args.in_channel, args.display, save=args.save, load=args.load)

get_options()
