from typing import *


class GuidanceIntervalSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance with interval.
    """

    def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength,  cfg_interval, extract_f = False, **kwargs):
        if extract_f:
                pred, feature_list = super()._inference_model(model, x_t, t, cond,extract_f = extract_f, **kwargs)
                neg_pred, feature_list_neg = super()._inference_model(model, x_t, t, neg_cond, extract_f = extract_f, **kwargs)

                fin_feature_list = []
                for i in range(len(feature_list)):
                    fin_feature_list.append(feature_list[i] * (1 + cfg_strength) - feature_list_neg[i] * cfg_strength)

                return (1 + cfg_strength) * pred - cfg_strength * neg_pred, fin_feature_list
        else:
            pred, = super()._inference_model(model, x_t, t, cond,extract_f = extract_f, **kwargs)
            neg_pred = super()._inference_model(model, x_t, t, neg_cond, extract_f = extract_f, **kwargs)
            return (1 + cfg_strength) * pred - cfg_strength * neg_pred

        # if cfg_interval[0] <= t <= cfg_interval[1]:
        #     if extract_f:
        #         pred, feature_list = super()._inference_model(model, x_t, t, cond,extract_f = extract_f, **kwargs)
        #         neg_pred, _ = super()._inference_model(model, x_t, t, neg_cond, extract_f = extract_f, **kwargs)
        #         return (1 + cfg_strength) * pred - cfg_strength * neg_pred, feature_list
        #     else:
        #         pred, = super()._inference_model(model, x_t, t, cond,extract_f = extract_f, **kwargs)
        #         neg_pred = super()._inference_model(model, x_t, t, neg_cond, extract_f = extract_f, **kwargs)
        #         return (1 + cfg_strength) * pred - cfg_strength * neg_pred
        # else:
        #     return super()._inference_model(model, x_t, t, cond,extract_f = extract_f, **kwargs)
