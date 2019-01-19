import argparse
from train import train
#from evaluate import evaluate
#from validate import validate

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
    parser.add_argument("--load", action = 'store_true')
    parser.add_argument("--eval", action = 'store_true')
    parser.add_argument("--validate", action = 'store_true')
    args = parser.parse_args()
    train(args.epochs, args.lr, args.n_class, args.in_channel, args.display, save=args.save, load=args.load)
    #if args.eval:
    #    evaluate()
    #elif args.validate:
    #    validate(args.display)
    #else:
    #    train(args.epochs, args.lr, args.n_class, args.in_channel, args.display, save=args.save, load=args.load)

get_options()
