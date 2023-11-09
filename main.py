import parameters
import tools
from model import *
import torch
import argparse

def main():
    #Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n_tokens', 
        type = int,
        help = 'size of text to produce (nb of characters)',
        choices = range(1, 3001),
        default = 400,
        )
    parser.add_argument(
        '--context', 
        type = str,
        help = 'context for the created sequence (max 300 chars)',
        default = '\n',
        )
    
    args = parser.parse_args()
    max_tokens = args.n_tokens
    context = args.context + '\n'


    #Loading model
    weights_file = 'model_1.33.pth'
    model = torch.load(weights_file, map_location=parameters.device)
    model.eval()


    #Creating a Friends Sequence
    input = torch.tensor(tools.encode(context), dtype=torch.long).view(1,len(context)).to(parameters.device)
    print(tools.decode(model.generate(idx = input, max_new_tokens=max_tokens)[0].tolist()))


if __name__ == "__main__":
    main()