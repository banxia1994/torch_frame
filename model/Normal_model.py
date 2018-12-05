import torch
from .models import BaseModel
from .model import MobileFaceNet
class normalModel(BaseModel):
    def __init__(self,opt):
        super(normalModel,self).__init__(opt)
        self._name = opt.model

        # create networks
        self._init_create_networks()

        # init train variables
        if self._is_train:
            self._init_train_vars()

        # prefetch variables
        self._init_prefetch_inputs()

        # init
        self._init_losses()

    def _init_create_networks(self):
        self._model = MobileFaceNet(512)

        if len(self._gpu_ids) > 1:
            self._model = torch.nn.DataParallel(self._model,device_ids = self._gpu_ids)
        self._model.cuda()

    def _init_train_vars(self):
        self._current_lr = self._opt._current_lr

        self._optimizer = torch.optim.Adam(self._model.parameters(),lr=self._current_lr)

    def _init_prefetch_inputs(self):
        pass

    def _init_losses(self):
        self._criterion = torch.nn.CrossEntropyLoss()

    def set_input(self, input):
        self.img = input['image']
        self.label = input['label']

    def set_train(self):
        self._model.train()
        self._is_train = True

    def set_eval(self):
        self._model.eval()
        self._is_train = False

    #def forward(self, keep_data_for_visuals=False):

    def optimize_parameters(self):
        if self._is_train:
            out = self._model(self.img)
            loss = self._criterion(out,self.label)

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

    def save(self,epoch):
        # save networks
        self._save_network(self._model,'model',epoch)

        # save optimizers
        self._save_optimizer(self._optimizer,'opti',epoch)

    def load(self):
        load_epoch = self._opt.load_epoch

        #load model
        self._load_network(self._model,'model',load_epoch)

        if self._is_train:
            self._load_network(self._model,'model',load_epoch)

            self._load_optimizer(self._optimizer,'opti',load_epoch)

    def update_learning_rate(self):
        lr_decay = self._opt.lr / self._opt.nepochs_decay
        self._current_lr -= lr_decay

        for param_group in self._optimizer.param_grops:
            param_group['lr'] = self._current_lr

        print ('update learning rate : %f -> %f ' % (self._current_lr + lr_decay, self._current_lr))