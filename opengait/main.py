
import os
import argparse
import torch
import torch.nn as nn
from modeling import models
from utils import config_loader, get_ddp_module, init_seeds, params_count, get_msg_mgr, mkdir

# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser(description='Main program for opengait.')
parser.add_argument('--local_rank', type=int, default=0,
                    help="passed by torch.distributed.launch module")
parser.add_argument('--cfgs', type=str,
                    default='./config/stdn/casiab/stdn-ca-1.yaml', help="path of config file")
parser.add_argument('--phase', default='test',
                    choices=['train', 'test'], help="choose train or test phase")
parser.add_argument('--log_to_file', action='store_true',
                    help="log to file, default path is: output/<dataset>/<model>/<save_name>/<logs>/<Datetime>.txt")
parser.add_argument('--iter', default=0, help="iter to restore")
opt = parser.parse_args()


def initialization(cfgs, training):
    msg_mgr = get_msg_mgr()
    engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
    output_path = os.path.join('output/', cfgs['data_cfg']['dataset_name'],
                               cfgs['model_cfg']['model'], cfgs['save_name'])
    
    if training:
        msg_mgr.init_manager(output_path, opt.log_to_file, engine_cfg['log_iter'], engine_cfg['total_iter'],
                             engine_cfg['restore_hint'] if isinstance(engine_cfg['restore_hint'], (int)) else 0)
    else:
        msg_mgr.init_logger(output_path, opt.log_to_file)

    msg_mgr.log_info(cfgs['discription'])
    msg_mgr.log_info(engine_cfg)

    seed = torch.distributed.get_rank()
    init_seeds(seed)


def run_model(cfgs, training):
    msg_mgr = get_msg_mgr()
    model_cfg = cfgs['model_cfg']
    msg_mgr.log_info(model_cfg)
    Model = getattr(models, model_cfg['model'])
    model = Model(cfgs, training)
    if training and cfgs['trainer_cfg']['sync_BN']:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if cfgs['trainer_cfg']['fix_BN']:
        model.fix_BN()
    model = get_ddp_module(model)
    msg_mgr.log_info(params_count(model))
    msg_mgr.log_info("Model Initialization Finished!")

    if training:
        Model.run_train(model)
    else:
        # Model.toy_test(dir_path='/ai/tps/project/Work-1/visual/sils/0', model=model)
        Model.run_test(model)


if __name__ == '__main__':

    # # for debug
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    # rank = opt.local_rank
    # size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    # torch.distributed.init_process_group('nccl', init_method='env://', rank=rank, world_size=size)

    # # for use
    torch.distributed.init_process_group('nccl', init_method='env://')
    
    if torch.distributed.get_world_size() != torch.cuda.device_count():
        raise ValueError("Expect number of availuable GPUs({}) equals to the world size({}).".format(
            torch.cuda.device_count(), torch.distributed.get_world_size()))
    cfgs = config_loader(opt.cfgs)
    save_path = os.path.join('output/', cfgs['data_cfg']['dataset_name'], cfgs['model_cfg']['model'], cfgs['save_name'])
    if torch.distributed.get_rank() == 0:
        mkdir(save_path)
    os.system('cp {} {}'.format(opt.cfgs, save_path))
    if opt.iter != 0:
        cfgs['evaluator_cfg']['restore_hint'] = int(opt.iter)
        cfgs['trainer_cfg']['restore_hint'] = int(opt.iter)

    training = (opt.phase == 'train')
    initialization(cfgs, training)
    run_model(cfgs, training)
