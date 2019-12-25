import argparse, time, os
import imageio
import torch
import options.options as option
from math import *
from utils import util
from solvers import create_solver
from data import create_dataloader
from data import create_dataset
import runway
from runway.data_types import *
import numpy as np
import common

opt = None
@runway.setup(options={"scale" : file(is_directory="True")})
def setup(opts):
    global opt
    model_scale = opts["scale"]
    model = model_scale + "/" + "model.pth"     
    config = model_scale + "/" + "config.json"

    opt = option.parse(config)
    opt = option.dict_to_nonedict(opt)

    solver = create_solver(opt, model)

    return solver

command_inputs = {"input_image" : image}
command_outputs = {"output_image" : image}


@runway.command("upscale_image", inputs=command_inputs, outputs=command_outputs, description="Upscales Image as per configuration")
def upscale_image(solver, inputs):
    print('Start Process')

    lr = np.array(inputs["input_image"])
    lr_tensor = common.np2Tensor([lr], opt['rgb_range'])[0]
    lr_tensor = lr_tensor.unsqueeze(0)

    solver.feed_data(lr_tensor, need_HR=False)
    solver.test()

    visuals = solver.get_current_visual(need_HR=False)

    img = visuals['SR']
    
    print("Finished Upscaling!")

    return {"output_image" : img}

if __name__ == "__main__":
    runway.run()
