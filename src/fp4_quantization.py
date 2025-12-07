import torch

def fp4_121_positive(x: torch.Tensor, stochastic_rounding: bool = False) -> torch.Tensor:
    """
    Quantizes positive values to FP4 (E2M1) format.
    Values are assumed to be scaled to the range [0, 6.0].
    FP4 (E2M1) values: 0, 0.5, 1, 1.5, 2, 3, 4, 6
    """
    if stochastic_rounding:
        noise = torch.rand_like(x) - 0.5
        step1 = torch.round(2.0 * x + noise) / 2.0
        step2 = torch.round(x + noise)
        step3 = 2.0 * torch.round(x / 2.0 + noise)
    else:
        step1 = torch.round(2.0 * x) / 2.0
        step2 = torch.round(x)
        step3 = 2.0 * torch.round(x / 2.0)
    
    mask1 = x < 2.0
    mask2 = x < 4.0

    return step1 * mask1 + step2 * (~mask1) * mask2 + step3 * (~mask1) * (~mask2)

def ue5m3(x: torch.Tensor) -> torch.Tensor:
    """
    Simulates UE5M3 quantization (unsigned E5M3).
    """
    # NOTE: Assume that array values are in [0, 114688]. (14*2**13 = 114688)
    mask = x <= 2**(-17)
    x_1 = x * mask
    x_2 = x * (~mask) + torch.ones_like(x) * mask

    x_1 = torch.round(x_1 / 2**(-17)) * (2**(-17))

    e = torch.floor(torch.log2(x_2)) - 3
    s = 2**e
    x_2 = torch.round(x_2 / s) * s

    return x_1 * mask + x_2 * (~mask)

FP8_E4M3_MAX = 240.0

def fp4_121_scaled(x: torch.Tensor, 
                   stochastic_rounding: bool = False, 
                   scale_format: str = 'e8m0') -> torch.Tensor:
    fp4_121_max = 6.0
    sign = x.sign()
    x_abs = x.abs()
    
    if scale_format == 'e8m0':
        # Power-of-2 scaling
        scale = torch.pow(2.0, torch.floor(torch.log2(fp4_121_max / x_abs.max(dim=-1, keepdim=True)[0])))
    
    elif scale_format == 'e4m3':
        # Simulated E4M3 scaling (replacing HPU op)
        # E4M3 has max value 240.0 (standard) or 448 (extended). 
        # The original code used 240.0.
        # We will just use float32/bf16 here as a fallback for the scale itself, 
        # or implement a simple clamp if we really wanted to simulate the range.
        # For this toy implementation, we'll treat it similar to bf16 but maybe clamp the range if needed.
        # Original code logic:
        nvfp4_max = fp4_121_max * FP8_E4M3_MAX
        scale_per_t = x_abs.max() / nvfp4_max
        x_abs_scaled = x_abs / scale_per_t

        scale_per_b = x_abs_scaled.max(dim=-1, keepdim=True)[0]
        
        # HPU code was:
        # down_cast = torch.ops.hpu.cast_to_fp8_v2(fp4_121_max / scale_per_b, ...)
        # We will skip the explicit FP8 cast and just use the value, 
        # effectively simulating infinite precision for the scale (or BF16/FP32).
        # If we wanted to be strict, we'd quantize `fp4_121_max / scale_per_b` to E4M3.
        
        # Simplified:
        scale_per_b = fp4_121_max / scale_per_b
        # Clamp to avoid inf/nan issues similar to original
        scale_per_b = torch.where((0 < scale_per_b) * (scale_per_b < float('inf')), scale_per_b, 1.0)

        x_fp4_abs = fp4_121_positive(x_abs_scaled * scale_per_b, stochastic_rounding) / scale_per_b

        return sign * x_fp4_abs * scale_per_t
    
    elif scale_format == 'ue5m3':
        UE5M3_MAX = 114688.0
        nvfp4_max = fp4_121_max * UE5M3_MAX
        scale_per_t = x_abs.max() / nvfp4_max
        x_abs_scaled = x_abs / scale_per_t

        scale_per_b = x_abs_scaled.max(dim=-1, keepdim=True)[0]

        scale_per_b = ue5m3(fp4_121_max / scale_per_b)
        
        scale_per_b = torch.where((0 < scale_per_b) * (scale_per_b < float('inf')), scale_per_b, 1.0)

        x_fp4_abs = fp4_121_positive(x_abs_scaled * scale_per_b, stochastic_rounding) / scale_per_b

        return sign * x_fp4_abs * scale_per_t

    else: # scale_format == 'bf16' or others
        scale = fp4_121_max / x_abs.max(dim=-1, keepdim=True)[0]
        scale = torch.where((0 < scale) * (scale < float('inf')), scale, 1.0)
        x_fp4_abs = fp4_121_positive(x_abs * scale, stochastic_rounding) / scale
        return sign * x_fp4_abs

    # Fallback for e8m0 path which didn't return in the if block
    scale = torch.where((0 < scale) * (scale < float('inf')), scale, 1.0)
    x_fp4_abs = fp4_121_positive(x_abs * scale, stochastic_rounding) / scale
    return sign * x_fp4_abs


def fake_quant_fp4(x: torch.Tensor, 
                   stochastic_rounding: bool = False, 
                   dim: int = -1, 
                   format: str = 'fp4_e2m1',
                   block_size: int = 32, 
                   scale_format: str = 'e8m0',
                   grid: bool = False) -> torch.Tensor:
    """
    Applies fake FP4 quantization to the input tensor.
    """
    shape = x.shape
    # Flatten to apply block-wise quantization
    if grid:
        assert len(shape) == 2, 'grid enabled for 2d tensors only'
        # Block-wise quantization for 2D matrices (e.g. weights)
        # Reshape to (rows//block, block, cols//block, block) -> permute -> flatten
        # This groups blocks of (block_size, block_size) together
        x = x.reshape(shape[0] // block_size, block_size, shape[1] // block_size, block_size).permute(0, 2, 1, 3).reshape(-1, block_size * block_size)
    else:
        # Simple block-wise quantization along the last dimension
        x = x.reshape(-1, block_size)
    
    x = fp4_121_scaled(x, stochastic_rounding, scale_format)
    
    if grid:
        x = x.reshape(shape[0] // block_size, shape[1] // block_size, block_size, block_size).permute(0, 2, 1, 3).reshape(shape)
    else:
        x = x.reshape(shape)
    
    return x
