import torch
import torch.nn as nn
import hashlib
import itertools

class SeededBatchNorm2d(nn.Module):
    
    def __init__(self, num_features, seed=0):
        super(SeededBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.hashed_seed = {}
        m = hashlib.sha256()
        m.update(str(seed).encode('utf-8'))
        self.hash = m.digest()
        self.seed_num = 0
        self.hashed_seed[self.hash] = "bn_" + str(self.seed_num)
        self.seed_num += 1
        if torch.cuda.is_available():
            self.__setattr__(self.hashed_seed[self.hash], nn.BatchNorm2d(num_features).cuda())
        else:
            self.__setattr__(self.hashed_seed[self.hash], nn.BatchNorm2d(num_features))
        
    def forward(self, x):
        self.__getattr__(self.hashed_seed[self.hash]).training = self.training
        return self.__getattr__(self.hashed_seed[self.hash]).forward(x)
        
    def reset_seed(self, seed):
        m = hashlib.sha256()
        m.update(str(seed).encode('utf-8'))
        h = m.digest()
        if not h in self.hashed_seed:
            self.hashed_seed[h] = "bn_" + str(self.seed_num)
            self.seed_num += 1
            if torch.cuda.is_available():
                self.__setattr__(self.hashed_seed[h], nn.BatchNorm2d(self.num_features).cuda())
            else:
                self.__setattr__(self.hashed_seed[h], nn.BatchNorm2d(self.num_features))
        self.hash = h
        
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        destination._metadata[prefix[:-1]] = dict(version=self._version)
        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param if keep_vars else param.data
        for name, buf in self._buffers.items():
            if buf is not None:
                destination[prefix + name] = buf if keep_vars else buf.data
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination, prefix + name + '.', keep_vars=keep_vars)
        destination[prefix + 'hashed_seed'] = self.hashed_seed
        return destination

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        
        self.hashed_seed = state_dict[prefix + 'hashed_seed']
        self.seed_num = len(self.hashed_seed)
        for h in self.hashed_seed:
            if not hasattr(self, self.hashed_seed[h]):
                if torch.cuda.is_available():
                    self.__setattr__(self.hashed_seed[h], nn.BatchNorm2d(self.num_features).cuda())
                else:   
                    self.__setattr__(self.hashed_seed[h], nn.BatchNorm2d(self.num_features))

        local_name_params = itertools.chain(self._parameters.items(), self._buffers.items())
        local_state = {k: v.data for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]

                # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
                if len(param.shape) == 0 and len(input_param.shape) == 1:
                    input_param = input_param[0]

                if input_param.shape != param.shape:
                    # local shape should match the one in checkpoint
                    error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                                      'the shape in current model is {}.'
                                      .format(key, input_param.shape, param.shape))
                    continue

                if isinstance(input_param, Parameter):
                    # backwards compatibility for serialized parameters
                    input_param = input_param.data
                try:
                    param.copy_(input_param)
                except Exception:
                    error_msgs.append('While copying the parameter named "{}", '
                                      'whose dimensions in the model are {} and '
                                      'whose dimensions in the checkpoint are {}.'
                                      .format(key, param.size(), input_param.size()))
            elif strict:
                missing_keys.append(key)

        if strict:
            for key, input_param in state_dict.items():
                if key.startswith(prefix):
                    input_name = key[len(prefix):]
                    input_name = input_name.split('.', 1)[0]  # get the name of param/buffer/child
                    if input_name not in self._modules and input_name not in local_state:
                        unexpected_keys.append(key)

