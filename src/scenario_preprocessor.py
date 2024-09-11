import yaml
import argparse as arg
from preprocessor_utils import preprocess_data, preprocess_maps


# 
def main():
    # get the arguments
    parser = arg.ArgumentParser()
    parser.add_argument('--config', type=str, default='../config/default.yaml', help='Path to the yaml file')
    args = parser.parse_args()
    
    # config file is a yaml file with a dictionary e.g. src_dir: /path/to/data
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    
    # read in keys
    src_dir = config['src_dir']
    output_dir = config['output_dir']
    interpolate = config['interpolate']
    
    preprocess_data(src_dir, output_dir, interpolate)
    
    if config['create_maps']:
        preprocess_maps(src_dir, output_dir, interpolate)
    

if __name__ == "__main__":
    main()
