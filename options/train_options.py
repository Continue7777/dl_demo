from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # exp_config
        self.parser.add_argument('--config_name', type=str,default='ConfigBase', help='config')

        # for displays
        self.parser.add_argument('--epoch_num', type=int, default=1000, help='train epoch nums')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=10000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=10000, help='frequency of saving checkpoints at the end of epochs')

        # for training
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--niter', type=int, default=10000, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=10000, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--beta2', type=float, default=0.999, help='smooth term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        # self.parser.add_argument('--ttur_mult', type=float, default=1, help='ttur_mult for D')


        self.isTrain = True

if __name__ == '__main__':
    opt = TrainOptions()
    # opt.parse()
    opt.initialize()
    # opt = opt.parser.parse_knowpan_args(args=[])[0]
    opt = vars(opt.parser.parse_args())
    # args = vars(opt)

    print(opt)
    print(opt.gan_mode)
    # print(args)
    # print('------------ Options -------------')
    # for k, v in sorted(args.items()):
    #     print('%s: %s' % (str(k), str(v)))
    # print('-------------- End ----------------')