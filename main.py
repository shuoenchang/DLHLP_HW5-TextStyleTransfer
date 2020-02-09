import torch
import time
from data import load_dataset
from models import StyleTransformer, Discriminator
from train import train, auto_eval
import argparse

class Config():
    data_path = './data/yelp/'
    log_dir = 'runs/exp'
    save_path = './save'
    pretrained_embed_path = './embedding/'
    #device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')
    discriminator_method = 'Multi' # 'Multi' or 'Cond'
    load_pretrained_embed = False
    min_freq = 3
    max_length = 16
    embed_size = 256
    d_model = 256
    h = 4
    num_styles = 2
    num_classes = num_styles + 1 if discriminator_method == 'Multi' else 2
    num_layers = 4
    batch_size = 64
    lr_F = 0.0001
    lr_D = 0.0001
    L2 = 0
    iter_D = 10
    iter_F = 5
    F_pretrain_iter = 500
    log_steps = 5
    eval_steps = 25
    learned_pos_embed = True
    dropout = 0
    drop_rate_config = [(1, 0)]
    temperature_config = [(1, 0)]

    slf_factor = 0.25
    cyc_factor = 0.5
    adv_factor = 1

    inp_shuffle_len = 0
    inp_unk_drop_fac = 0
    inp_rand_drop_fac = 0
    inp_drop_prob = 0


def main(args):
    #config = Config()
    train_iters, dev_iters, test_iters, vocab = load_dataset(args)
    print('Vocab size:', len(vocab))
    model_F = StyleTransformer(args, vocab).to(args.device)
    model_D = Discriminator(args, vocab).to(args.device)
    print(args.discriminator_method)
    
    train(args, vocab, model_F, model_D, train_iters, dev_iters, test_iters)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", dest="data_path", default="./data/yelp/")
    parser.add_argument("-log_dir", dest="log_dir", default="runs/exp")
    parser.add_argument("-save_path", dest="save_path", default="./save")
    parser.add_argument("-pretrained_embed_path", dest="pretrained_embed_path", default="./embedding/")
    parser.add_argument("-discriminator_method", dest="discriminator_method", default="Multi")
    parser.add_argument("--load_pretrained_embed", dest="load_pretrained_embed", action="store_true")
    parser.add_argument("-min_freq", dest="min_freq", default=3, type=int)
    parser.add_argument("-max_length", dest="max_length", default=16, type=int)
    parser.add_argument("-embed_size", dest="embed_size", default=256, type=int)
    parser.add_argument("-d_model", dest="d_model", default=256, type=int)
    parser.add_argument("-head", dest="h", default=4, type=int)
    parser.add_argument("-num_styles", dest="num_styles", default=2, type=int)
    parser.add_argument("-num_layers", dest="num_layers", default=4, type=int)
    parser.add_argument("-batch_size", dest="batch_size", default=64 , type=int)
    parser.add_argument("-lr_F", dest="lr_F", default=1e-4 , type=float)
    parser.add_argument("-lr_D", dest="lr_D", default=1e-4 , type=float)
    parser.add_argument("-L2", dest="L2", default=0.0 , type=float)
    parser.add_argument("-iter_D", dest="iter_D", default=10 , type=int)
    parser.add_argument("-iter_F", dest="iter_F", default=5 , type=int)
    parser.add_argument("-F_pretrain_iter", dest="F_pretrain_iter", default=500 , type=int)
    parser.add_argument("-log_steps", dest="log_steps", default=5 , type=int)
    parser.add_argument("-eval_steps", dest="eval_steps", default=25 , type=int)
    parser.add_argument("-learned_pos_embed", dest="learned_pos_embed", default=True, type=bool)
    parser.add_argument("-dropout", dest="dropout", default=0 , type=int)
    
    parser.add_argument("-slf_factor", dest="slf_factor", default=0.25, type=float)
    parser.add_argument("-cyc_factor", dest="cyc_factor", default=0.5, type=float)
    parser.add_argument("-adv_factor", dest="adv_factor", default=1, type=float)
    
    parser.add_argument("-inp_shuffle_len", dest="inp_shuffle_len", default=0, type=int)
    parser.add_argument("-inp_unk_drop_fac", dest="inp_unk_drop_fac", default=0, type=int)
    parser.add_argument("-inp_rand_drop_fac", dest="inp_rand_drop_fac", default=0, type=int)
    parser.add_argument("-inp_drop_prob", dest="inp_drop_prob", default=0, type=int)
    

    args = parser.parse_args()
    args.drop_rate_config = [(1, 0)]
    args.temperature_config = [(1, 0)]
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.num_classes = args.num_styles + 1 if args.discriminator_method == 'Multi' else 2
    
    main(args)
