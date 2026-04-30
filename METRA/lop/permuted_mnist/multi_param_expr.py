import sys
import json
import copy
import argparse
from pathlib import Path
import shutil
from tqdm import tqdm
from lop.utils.miscellaneous import get_configurations


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-c',
        help="Path of the file containing the parameters of the experiment",
        type=str,
        default='cfg/a.json'
    )
    args = parser.parse_args(arguments)
    cfg_file = args.c

    with open(cfg_file, 'r') as f:
        params = json.load(f)

    list_params, hyper_param_settings = get_configurations(params=params)

    # Create temp cfg directory
    temp_cfg_dir = Path('temp_cfg')
    temp_cfg_dir.mkdir(parents=True, exist_ok=True)

    # Reset output data directory
    data_dir = Path(params['data_dir'])
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    """
    Set and write all the parameters for the individual config files
    """
    for setting_index, param_setting in enumerate(hyper_param_settings):
        new_params = copy.deepcopy(params)

        for idx, param in enumerate(list_params):
            new_params[param] = param_setting[idx]

        new_params['index'] = setting_index
        setting_data_dir = data_dir / str(setting_index)
        new_params['data_dir'] = str(setting_data_dir).replace('\\', '/') + '/'

        # Make the data directory for this setting
        setting_data_dir.mkdir(parents=True, exist_ok=True)

        for idx in tqdm(range(params['num_runs'])):
            new_params['data_file'] = str(setting_data_dir / str(idx)).replace('\\', '/')

            new_cfg_file = temp_cfg_dir / f'{setting_index * params["num_runs"] + idx}.json'
            with open(new_cfg_file, 'w+') as f:
                json.dump(new_params, f, sort_keys=False, indent=4)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))