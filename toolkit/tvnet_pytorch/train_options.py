import argparse

class arguments():
    def __init__(self):
        self.argparser = argparse.ArgumentParser(description='abc')
        self.initialize()

    def initialize(self):
        self.argparser.add_argument('--frame_dir', type=str, default='frame', help='path to frames')
        self.argparser.add_argument('--img_save_dir', type=str, default='results', help='path to storage generated feature maps if needed')
        self.argparser.add_argument('--n_epoch', type=int, default=1000, help='number of epochs')
        self.argparser.add_argument('--n_threads', type=int, default=1, help='number of threads for dataloader')
        self.argparser.add_argument('--batch_size', type=int, default=1, help='just batch size')
        self.argparser.add_argument('--learning_rate', type=float, default=0.005, help='learning rate')
        self.argparser.add_argument('--is_shuffle', type=bool, default=False, help='Do shuffle during loading data or not')
        self.argparser.add_argument('--visualize', type=bool, default=True, help='storage the flow in image type')        
        self.argparser.add_argument('--data_size', help='input data size',default=[1,3,426,1])
        self.argparser.add_argument('--zfactor', type=float, default=0.5, help='factor for building the image piramid')
        self.argparser.add_argument('--max_nscale', type=int, default=1, help='maximum number of scales for image piramid')
        self.argparser.add_argument('--tau', type=float, default=0.25, help='time step')
        self.argparser.add_argument('--lbda', type=float, default=0.15, help='weight parameter for the data term')
        self.argparser.add_argument('--theta', type=float, default=0.3, help='weight parameter for (u - v)^2')
        self.argparser.add_argument('--n_warps', type=int, default=1, help='number of warpings per scale')
        self.argparser.add_argument('--n_iters', type=int, default=50, help='maximum number of iterations for optimization')
        self.argparser.add_argument('--demo', help="just demo with original weights", action="store_true")
        self.argparser.add_argument('--dataset', type=str,help='datasets',default='OTB100')
        self.argparser.add_argument('--vis', action="store_true")
        self.argparser.add_argument('--gpu', action="store_true")
        #
        self.argparser.add_argument('--config', default='', type=str,help='config file')
        self.argparser.add_argument('--snapshot', default='', type=str,help='snapshot of models to eval')
        self.argparser.add_argument('--video', default='', type=str,help='eval one special video')
        self.argparser.add_argument('--max_num', type=int,default=10)
        self.argparser.add_argument('--interval', type=int,default=-1,help='-1:apt,>0:attack with fixed interval, inf:only use the perturation from the first frame')
        self.argparser.add_argument('--opt_flow', action='store_true', help='whether using optical flow')
        self.argparser.add_argument('--apts', action='store_true',help='whether attacking apts')

    def parse(self):
        self.args = self.argparser.parse_args()
        return self.args
