import logging
import os
import time
import json

from datetime import datetime
from tqdm import tqdm

from src.methods import *
from src.utils import get_args
from src.utils.conf import cfg, load_cfg_fom_args
from src.data.data import load_dataset
from src.models.base_model import BaseModel
from src.models.convnext import ConvNeXt_para


logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(cfg):
########################################################################################################################
    model_dict = {}
    arch_name = cfg.MODEL.ARCH

    model_args = dict(depths=[3, 4, 30, 3], dims=[384, 768, 1536, 3072], norm_eps=1e-5)
    model = ConvNeXt_para(3, 1000, **dict(model_args))
    model = BaseModel(model, arch_name)

    model_load = torch.load(cfg.CHEAKPOINT)

    for nm, para in model_load.items():
        if (nm.split(".")[0]) == 'model':
            model_dict[nm[13:]] = para

    model.load_state_dict(model_dict)

    model.to(device)
    model.eval()

    return model
#########################################################################################################################

def main():
    args = get_args()
    args.output_dir = args.output_dir if args.output_dir else 'online_evaluation'
    load_cfg_fom_args(args.cfg, args.output_dir)
    logger.info(cfg)
    start_time = time.time()
    dom_names_all = cfg.CORRUPTION.TYPE
    logger.info(f"Using the following domain sequence: {dom_names_all}")

    # prevent iterating multiple times over the same data in the mixed_domains setting
    dom_names_loop = dom_names_all

    ###################################################################################################################
    annotations = json.load(open(cfg.ANNOTATION_PATH))
    image_list = annotations["images"]
    indices_in_1k = [d['id'] for d in annotations['categories']]
    ###################################################################################################################


    model = load_model(cfg)

    results = {}

    testset, test_loader = load_dataset(cfg.CORRUPTION.DATASET, cfg.DATA_DIR,
                                        cfg.TEST.BATCH_SIZE,
                                        split='all',
                                        adaptation=cfg.MODEL.ADAPTATION,
                                        workers=min(cfg.TEST.NUM_WORKERS, os.cpu_count()),
                                        ckpt=os.path.join(cfg.CKPT_DIR, 'Datasets'),
                                        num_aug=cfg.TEST.N_AUGMENTATIONS,
                                        model_arch=cfg.MODEL.ARCH)

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            imgs, labels = data[0], data[1]
            output, logit = model([img.to(device) for img in imgs], return_feats = True) if isinstance(imgs, list) else model(imgs.to(device), return_feats = True)

            logit = logit[:, indices_in_1k]

            for j in range(len(logit)):
                data_num = i * cfg.TEST.BATCH_SIZE + j
                image_name = test_loader.dataset.samples[data_num][0].split("/")[-1]
                if image_name in image_list:
                    results[image_name] = indices_in_1k[logit[j].argmax()]

    file_path = os.path.join(args.output_dir, datetime.now().strftime(f'prediction-%m-%d-%Y-%H:%M:%S.json'))
    with open(file_path, 'w') as outfile:
        json.dump(results, outfile)

if __name__ == "__main__":
    main()
