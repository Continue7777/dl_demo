import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter


class Visualizer():
    def __init__(self, opt):
        self.name = opt.name
        self.log_dir = os.path.join(opt.savePath, opt.name)
        self.writer = SummaryWriter(self.log_dir)

        self.log_name = os.path.join(self.log_dir, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors, step):
        for tag, value in errors.items():
            self.writer.add_scalar(tag, value, global_step=step)

    def plot_gradient(self, tag, grads, step):
        self.writer.add_scalar("grad_max/" + tag, torch.max(grads), global_step=step)
        self.writer.add_scalar("grad_std/" + tag, torch.std(grads), global_step=step)
        self.writer.add_scalar("grad_mean/" + tag, torch.mean(grads), global_step=step)
        self.writer.add_scalar("grad_min/" + tag, torch.min(grads), global_step=step)

    def plot_weight(self, tag, weights, step):
        self.writer.add_scalar("weights_max/" + tag, torch.max(weights), global_step=step)
        self.writer.add_scalar("weights_std/" + tag, torch.std(weights), global_step=step)
        self.writer.add_scalar("weights_mean/" + tag, torch.mean(weights), global_step=step)
        self.writer.add_scalar("weights_min/" + tag, torch.min(weights), global_step=step)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, itr, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, itr, t)
        for k, v in errors.items():
            if v != 0:
                message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, image, epoch, itr):
        if os.path.exists(os.path.join(self.log_dir, "images")) == False:
            os.mkdir(os.path.join(self.log_dir, "images"))
        image.save(os.path.join(self.log_dir, "images", str(epoch) + "_" + str(itr) + "_fake.png"))
        image.save(os.path.join(self.log_dir, "images", "latest_fake.png"))


