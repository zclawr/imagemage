import cmdparser
import util
import os
import numpy as np
from pathlib import Path
import glob

text_snippets = [
    "casting spells",
    "brewing potions",
    "killing imps",
    "cursing bloodlines",
    "performing astral necromancy",
    "crossing dimensional boundaries",
    "finding secret passageways",
    "listening to whispering souls",
    "scrying distant lands",
    "burning celestial poisons",
    "lighting everlasting flames",
    "peering beneath the veil",
    "summoning polar winds",
    ]

def save_results(cmd, output_dir, result, is_sound, sr):
    if len(result.shape) == 4: # Multi-image output needs separate saving
        for i in range(result.shape[0]): 
            util.save_output(cmd, output_dir, result[i], is_sound, sr, idx=i)
    else:
        util.save_output(cmd, output_dir, result, is_sound, sr)

if __name__ == '__main__':
    cmd, input_path, functions, strs, params, args = cmdparser.parse_cmd()
    output_dir = os.path.dirname(input_path)

    img, is_sound, sr = util.load_input_file(input_path)
    print(f'#~' * 50)
    print(f'o*' * 50)
    print()
    print(f'Imagemage is now {text_snippets[int(np.floor(len(text_snippets) * np.random.uniform()))]} with {input_path}')
    print()
    print(cmd)
    print()
    print(f'o*' * 50)
    print(f'#~' * 50)
    print()

    hash = util.short_deterministic_hash(cmd, util.HASH_LENGTH)
    hash_path = os.path.join(output_dir, hash)
    hash_file = glob.glob(f'{hash_path}*')
    # Checking if this has already been ran
    if len(hash_file) > 0 and not args.rerun:
        print(f'Found file generated with same command: {hash_path}')
    else:
        n = len(functions)
        start_idx = 0
        next = img
        if not args.rerun:
            for i in range(n, 0, -1): # Sweep backwards through functions to see if cmd up to that index has already been ran, if so use file
                cmd_i = cmdparser.format_cmd(input_path, strs[:i], params[:i])
                print(cmd_i)
                hash_i = util.short_deterministic_hash(cmd_i, util.HASH_LENGTH)
                hash_path_i = os.path.join(output_dir, f"{hash_i}")

                hash_files = glob.glob(f'{hash_path_i}*')
                if len(hash_files) > 0:
                    start_idx = i
                    print(f'Starting from existing files found with command: {cmd_i}')
                    tmp_next = []
                    for file in sorted(hash_files):
                        img, _, _ = util.load_input_file(file)
                        tmp_next.append(img)
                    tmp_next = np.array(tmp_next).squeeze()
                    next = tmp_next
                    break

        for i in range(start_idx, n):
            if len(params[i]) == 0:
                print(f'->{strs[i].split("(")[0]}()')
            elif len(params[i]) == 1:
                print(f'->{strs[i].split("(")[0]}{str(params[i]).split(',')[0]})')
            else:
                print(f'->{strs[i].split("(")[0]}{str(params[i]).strip()}')
            cmd_i = cmdparser.format_cmd(input_path, strs[:i+1], params[:i+1])

            func = functions[i]
            param = params[i]
            next = func(next, *param)

            if args.intermediate and i < n-1:
                print('Saving intermediate results for: ' + cmd_i)
                save_results(cmd_i, output_dir, next, is_sound, sr)

        save_results(cmd, output_dir, next, is_sound, sr)