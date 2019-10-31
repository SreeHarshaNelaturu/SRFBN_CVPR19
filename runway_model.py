import argparse, time, os
import imageio

import options.options as option
from math import *
from utils import util
from solvers import create_solver
from data import create_dataloader
from data import create_dataset
import common
import runway
from runway.data_types import *
import numpy as np

opt = None
@runway.setup(options={"upscaling_model" : file(extension=".pth", "config" : file(extension=".json"))})
def setup(opts):
    global opt
    model = opts["upscaling_model"]
    config = opts["config"]

    # ****** Model Parameters******
    scale = opt['scale']
    degrad = opt['degradation']
    network_opt = opt['networks']
    model_name = network_opt['which_model'].upper()
    if opt['self_ensemble']: model_name += 'plus'
    # *****************************

    opt = option.parse(config)
    opt = option.dict_to_nonedict(opt)

    solver = create_solver(opt)

    return solver

command_inputs = {"input_image" : image}
command_outputs = {"output_image" : image}


@runway.command("upscale_image", inputs=command_inputs, outputs=command_outputs, description="Upscales Image as per configuration")
def upscale_image(solver, inputs):
    total_time = []
    t0 = time.time()
    solver.test()
    t1 = time.time()
   #total_time.append((t1 - t0))
    # ******Command Definition******
    print('===> Start Test')
    print("==================================================")
    print("Method: %s || Scale: %d || Degradation: %s"%(model_name, scale, degrad))

    lr = np.array(inputs["input_image"])
    lr_tensor = common.np2Tensor(lr, opt['rgb_range'])[0]

    print("Time to Start: %.4f sec ." % (t1 - t0))

    solver.feed_data(lr_tensor, need_HR=False)

    #print("Time to Load: %.4f sec ." % (t1 - t0))

    visuals = solver.get_current_visual(need_HR=False)


    img = visuals['SR']
    print("Time to Complete: %.4f sec ." % (t1 - t0))
    print("==================================================")
    print("===> Finished !")

    return {"output_image" : img}

if __name__ == "__main__":
    runway.run()










