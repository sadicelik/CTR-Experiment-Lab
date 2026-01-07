# from functools import lru_cache

# from torch import nn


# class BaseModel(nn.Module):
#     def __init__(self, feature_map, **kwargs):
#         super(BaseModel, self).__init__()

#         self._feature_map = feature_map
#         self.feature_specs = self._feature_map.feature_specs

#     @lru_cache(maxsize=1)
#     def embed_params(self):
#         embed_params = []
#         for name, param in self.named_parameters():
#             if param.requires_grad:
#                 if "embedding_layer" in name and param.shape[-1] > 1:
#                     embed_params.append(param)
#         return embed_params

#     @lru_cache(maxsize=1)
#     def net_params(self):
#         net_params = []
#         for name, param in self.named_parameters():
#             if param.requires_grad:
#                 if "embedding_layer" not in name or param.shape[-1] == 1:
#                     net_params.append(param)
#         return net_params

#     @lru_cache(maxsize=1)
#     def get_feature_params_map(self):
#         feature_params_map = dict()
#         for feature, feature_spec in self.feature_specs.items():
#             feature_params_map[feature] = []
#             for name, param in self.named_parameters():
#                 if param.requires_grad:
#                     if "embedding_layer" in name and param.shape[-1] > 1:
#                         if feature in name:
#                             feature_params_map[feature].append(param)
#         return feature_params_map
