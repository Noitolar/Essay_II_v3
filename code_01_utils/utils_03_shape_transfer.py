import torch


def dv_t_e_to_dt_v_e(x: torch.Tensor, attn_mask: torch.Tensor, shapes: tuple):
    d, t, v, e = shapes
    assert x.shape == torch.Size([d * v, t, e])
    assert attn_mask.shape == torch.Size([d * v, t])
    x = x.reshape(d, v, t, e)
    x = torch.einsum("dvte->dtve", x)
    x = x.reshape(d * t, v, e)

    attn_mask = attn_mask.reshape(d, v, t)
    attn_mask = torch.einsum("dvt->dtv", attn_mask)
    attn_mask = attn_mask.reshape(d * t, v)

    return x, attn_mask


def dt_v_e_to_dv_t_e(x: torch.Tensor, attn_mask: torch.Tensor, shapes: tuple):
    d, t, v, e = shapes
    assert x.shape == torch.Size([d * t, v, e])
    assert attn_mask.shape == torch.Size([d * t, v])
    x = x.reshape(d, t, v, e)
    x = torch.einsum("dtve->dvte", x)
    x = x.reshape(d * v, t, e)

    attn_mask = attn_mask.reshape(d, t, v)
    attn_mask = torch.einsum("dtv->dvt", attn_mask)
    attn_mask = attn_mask.reshape(d * t, v)

    return x, attn_mask
