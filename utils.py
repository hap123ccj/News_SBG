import os,json

def save_args(args):
    if not os.path.exists(args.outputresult):
        os.mkdir(args.outputresult)
    with open(os.path.join(args.outputresult,'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)