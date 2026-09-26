import argparse
from datetime import datetime
import transforms
import transforms.compression
import transforms.convolutional
import transforms.binary
import transforms.cryptography
import transforms.nla
import transforms.periodic
import transforms.util
import transforms.probabilistic
import util
import os

## Command format:
## ---------------------
## {input name}:{transform(params)}->{transform(params)}->...
##
## Flags:
## --------
## -s : save command as cmd.txt
## -f {file path} : load command from file
# -i : save intermediate files
##
## Notes:
## --------
## - auto-detects file type and converts accordingly
## 
##
## Misc. commands
## ---------------------
## undo
## - deletes last batch

## TODO:
## - implement masking (e.g. via checker)
## - implement indexing of multi-output functions
## - implement last-command log for undo support
## - implement multi-line commands for re-routing of multi-output functions

cmd_map = {
    "svd": transforms.nla.low_rank_approximation_svd_per_channel,
    "svdk": transforms.nla.low_rank_approximation_svd,
    "ecb": transforms.cryptography.ecb,
    "ilpf": transforms.convolutional.ideal_lpf,
    "glpf": transforms.convolutional.gaussian_blur,
    "mod": transforms.util.modulo,
    "wavelet": transforms.nla.wavelet,
    "bitslice": transforms.binary.bitslice,
    "grad": transforms.convolutional.gradient_magnitude,
    "adaptive": transforms.convolutional.adaptive_local_noise_reduction,
    "inv": transforms.util.invert,
    "mono": transforms.util.mono,
    "chrotate": transforms.util.rotate_channels,
    "dct": transforms.compression.lossy_dct_transform,
    "add": transforms.util.add,
    "sub": transforms.util.subtract,
    "mul": transforms.util.multiply,
    "div": transforms.util.divide,
    "and": transforms.util.bitwise_and,
    "or": transforms.util.bitwise_or,
    "xor": transforms.util.bitwise_xor,
    "lshift": transforms.util.bitshift_left,
    "rshift": transforms.util.bitshift_right,
    "puzzle": transforms.periodic.puzzle,
    "uniform": transforms.probabilistic.uniform,
    "fbm": transforms.probabilistic.fracbm,
    "xflip": transforms.util.xflip,
    "yflip": transforms.util.yflip,
}

def write_cmd_file(cmd):
    file_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f"./{file_timestamp}.txt", "w") as file:
        file.write(cmd)

def load_cmd_file(path):
    with open(path, "r", encoding="utf-8") as file:
        cmd = file.read()
    return cmd

def parse_args():
    parser = argparse.ArgumentParser(description="=mm==-* THANK YOU FOR USING IMAGEMAGE <<<|:^O")
    parser.add_argument("command")

    parser.add_argument("-s", "--save", action="store_true")
    parser.add_argument("-i", "--intermediate", action="store_true")
    parser.add_argument("-r", "--rerun", action="store_true")
    parser.add_argument("-f", "--file", default='', type=str)

    args = parser.parse_args()

    cmd = args.command.strip()

    if(cmd == 'undo'):
        path = util.get_last_output()
        os.remove(path)

    if(args.save):
        write_cmd_file(cmd)

    if(args.file):
        cmd = load_cmd_file(args.file)

    return cmd, args

def get_cmd_functions(transforms):
    functions = []

    for transform in transforms:
        functions.append(cmd_map[transform.split("(")[0].strip()])

    return functions

def parse_cmd():
    raw_cmd, args = parse_args()
    input_path = raw_cmd.split(":")[0].strip()
    cmd = raw_cmd.split(":")[1].strip()
    transforms = cmd.split("->")

    params = []
    for transform in transforms:
        param_str = transform.split("(")[1].split(")")[0].strip()
        if(param_str==""):
            params.append(())
            continue
        param_tuple = tuple(map(int, param_str.split(",")))
        params.append(param_tuple)

    functions = get_cmd_functions(transforms)
    return raw_cmd, input_path, functions, transforms, params, args

def format_cmd(input_path, strs, params):
    cmd = input_path + ':'
    for i in range(len(strs)):
        if len(params[i]) <= 1:
            cmd += f'{strs[i].split('(')[0]}{str(params[i]).split(')')[0].split(',')[0].strip()})'
        else:
            cmd += f'{strs[i].split('(')[0]}{str(params[i]).strip()}'
        if i < len(strs)-1:
            cmd += '->'
    return cmd.strip()