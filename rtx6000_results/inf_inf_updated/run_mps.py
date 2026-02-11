import yaml
import itertools
import logging
import os
import copy
import subprocess
from pathlib import Path

mnames = {
    'resnet50': "ResNet50",
    'mobilenet_v2': "MobileNetV2",
    'resnet101': 'ResNet101',
    'bert': 'BERT',
}

def run(model0, model1, config, combination_name, times=1, start_id = 0):

    out_root = Path('results') / 'mps'
    configs_dir = out_root / 'configs'
    logs_dir = out_root / 'logs'
    configs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    config_path = configs_dir / f'gen_conf_{combination_name}.yaml'

    logging.info(f'dump config to {config_path}')
    with config_path.open('w') as file:
        yaml.dump(config, file)

    logging.info(f'training with this config {times} times')

    baseline_main = Path(os.path.expanduser('~')) / 'orion' / 'related' / 'baselines' / 'main.py'

    for i in range(start_id, start_id + times):
        log_path = logs_dir / f'log_{i}_{combination_name}.log'

        cmd = [
            'python3',
            str(baseline_main),
            '--config',
            str(config_path),
            '--log',
            str(log_path),
        ]
        subprocess.run(cmd, check=True)

        produced_json = Path(f'{log_path}.json')
        if not produced_json.exists():
            raise FileNotFoundError(f'Expected baseline output JSON not found: {produced_json}')

        final_json = out_root / f"{mnames[model0]}_{mnames[model1]}_{i}.json"
        produced_json.replace(final_json)



if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s: [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%d/%m/%Y %H:%M:%S',
        handlers=[
            # output to console
            logging.StreamHandler(),
        ]
    )
    with open('./config_files/mps/config.yaml', 'r') as file:
        default_full_config = yaml.load(file, Loader=yaml.FullLoader)

    # ----configuration region started----
    model0_mode = 'eval'
    model1_mode = 'eval'

    policy = 'MPS'

    models = ['resnet50', 'mobilenet_v2', 'resnet101', 'bert']
    combinations = itertools.product(models, models)
    times = 1
    start_id = 0
    distribution = 'poisson'

    for model0, model1 in combinations:
        default_full_config['shared_config']['distribution'] = copy.deepcopy(distribution)

        default_full_config['models']['model0']['name'] = model0
        default_full_config['models']['model0']['mode'] = model0_mode
        default_full_config['models']['model1']['name'] = model1
        default_full_config['models']['model1']['mode'] = model1_mode
        default_full_config['policy'] = policy
        # if model1 != model0:
        #     default_full_config[model1]['num_iterations'] = 10000000 # just a large number to avoid early stopping

        print(model0, model1, default_full_config)

        combination_name = f'{model0_mode}-{model0}{model1_mode}-{model1}'
        run(model0, model1, default_full_config, combination_name, times=times, start_id=start_id)
