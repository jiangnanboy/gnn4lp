from torch import optim

def define_optimizer_adam(model, lr=1e-3, weight_decay=0, betas=(0.9, 0.999), eps=1e-8, amsgrad=False):
    '''
    define optimizer function adam
    :param model:
    :param lr:
    :param weight_decay:
    :param betas:
    :param eps:
    :param amsgrad:
    :return:
    '''
    optimizer_adam = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps,amsgrad=amsgrad)
    return optimizer_adam

def define_optimizer_adamw(model, lr=1e-3, weight_decay=1e-2, betas=(0.9, 0.999), eps=1e-8, amsgrad=False):
    '''
    define optimizer function adamw
    :param model:
    :param lr:
    :param weight_decay:
    :param betas:
    :param eps:
    :param amsgrad:
    :return:
    '''
    optimizer_adamw = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps, amsgrad=amsgrad)
    return optimizer_adamw

def define_optimizer_sgd(model, lr=1e-3, momentum=0, weight_decay=0, nesterov=False, dampening=0):
    '''
    define optimizer function sgd
    :param model:
    :param lr:
    :param momentum:
    :param weight_decay:
    :param nesterov:
    :param dampening:
    :return:
    '''
    optimizer_sgd = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov, dampening=dampening)
    return optimizer_sgd

def define_optimizer_adagrad(model, lr=1e-2, lr_decay=0, weight_decay=0, eps=1e-10):
    '''
    define optimizer function adagrad
    :param model:
    :param lr:
    :param lr_decay:
    :param weight_decay:
    :param initial_accumulator_value:
    :param eps:
    :return:
    '''
    optimizer_adagrad = optim.Adagrad(model.parameters(), lr=lr, lr_decay=lr_decay, weight_decay=weight_decay, eps=eps)
    return optimizer_adagrad

def define_optimizer_rmsprop(model, lr=1e-2, weight_decay=0, momentum=0, alpha=0.99, eps=1e-8):
    '''
    define optimizer function rmsprop
    :param model:
    :param lr:
    :param weight_decay:
    :param momentum:
    :param alpha:
    :param eps:
    :return:
    '''
    optimizer_rmsprop = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, alpha=alpha, eps=eps)
    return optimizer_rmsprop

def define_optimizer_adadelta(model, lr=1.0, weight_decay=0):
    '''
    define optimizer function adadelta
    :param model:
    :param lr:
    :param weight_decay:
    :return:
    '''
    optimizer_adadelta = optim.Adadelta(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer_adadelta