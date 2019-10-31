from .SRSolver import SRSolver

def create_solver(opt, ckpt):
    if opt['mode'] == 'sr':
        solver = SRSolver(opt, ckpt)
    else:
        raise NotImplementedError

    return solver