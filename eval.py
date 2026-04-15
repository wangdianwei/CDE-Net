import os
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

from thop import profile
from thop import clever_format

from utils import pad_img
from option import opt
from model import Backbone


# ============================================================
# 配置：论文里写死的测速参数（你要求的固定值）
# ============================================================
INPUT_SIZE = (3, 480, 640)   # C,H,W
PAD_TO = 4                   # pad 到 4 的倍数（480/640 本来就满足）
WARMUP = 100                 # 预热次数（固定）
RUNS = 500                   # 正式运行次数（固定）
TRIALS = 5                   # 统计 mean±std 的重复次数（建议 5）


# ---------------------------
# ✅ 权重加载函数
# ---------------------------
def load_ckpt_safely(model, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]

    if isinstance(ckpt, dict):
        keys = list(ckpt.keys())
        if len(keys) > 0 and keys[0].startswith("module."):
            new_state = OrderedDict()
            for k, v in ckpt.items():
                new_state[k.replace("module.", "", 1)] = v
            ckpt = new_state

    model.load_state_dict(ckpt, strict=False)
    print("✅ 预训练权重加载成功（strict=False）:", ckpt_path)


# ---------------------------
# ✅ （可选）自动切换 Deploy（如果模型支持）
# ---------------------------
def try_switch_to_deploy(model: nn.Module):
    """
    如果 Backbone 实现了 re-parameterization / deploy 转换接口，则自动调用。
    不同代码库命名不同，这里做了常见方法名的兼容。
    """
    for fn_name in ["switch_to_deploy", "reparam", "repvgg_convert", "re_parameterize", "reparameterize"]:
        if hasattr(model, fn_name) and callable(getattr(model, fn_name)):
            print(f"✅ Switch to deploy by calling: model.{fn_name}()")
            getattr(model, fn_name)()
            return True
    return False


# ------------------------------
# ✅ CUDA Event 测速：输出 mean±std（论文格式）
# ------------------------------
@torch.no_grad()
def benchmark_latency_fps_cuda(
    network: nn.Module,
    input_size=(3, 480, 640),
    pad_to=4,
    warmup=100,
    runs=500,
    trials=5,
):
    """
    论文级测速：
    - warmup: 预热次数（固定 100）
    - runs: 每次 trial 的 forward 次数（固定 500）
    - trials: 重复次数，用于计算 mean±std（建议 5）
    输出：Latency(mean±std), FPS(mean±std)
    """
    network.eval()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，无法用 CUDA Event 测速。")

    # 固定随机输入（只创建一次，避免把创建/拷贝算进时间）
    x = torch.randn(1, *input_size, device="cuda")
    x = pad_img(x, pad_to)

    # Warm-up
    torch.cuda.synchronize()
    for _ in range(warmup):
        _ = network(x)
    torch.cuda.synchronize()

    lat_list = []
    fps_list = []

    # 多次 trial：每次用一个 start/end event 计总耗时，再除以 runs
    for _ in range(trials):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        starter.record()
        for _ in range(runs):
            _ = network(x)
        ender.record()
        torch.cuda.synchronize()

        elapsed_ms = starter.elapsed_time(ender)  # runs 次 forward 的总毫秒数
        lat_ms = elapsed_ms / runs
        fps = 1000.0 / lat_ms if lat_ms > 0 else 0.0

        lat_list.append(lat_ms)
        fps_list.append(fps)

    lat_arr = np.array(lat_list, dtype=np.float64)
    fps_arr = np.array(fps_list, dtype=np.float64)

    lat_mean = float(lat_arr.mean())
    lat_std = float(lat_arr.std(ddof=1)) if trials > 1 else 0.0
    fps_mean = float(fps_arr.mean())
    fps_std = float(fps_arr.std(ddof=1)) if trials > 1 else 0.0

    print("\n=== CUDA Event Benchmark (forward-only, fixed random input) ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Input: 1x{input_size[0]}x{input_size[1]}x{input_size[2]} (pad x{pad_to})")
    print(f"Setting: warmup={warmup}, runs={runs}, trials={trials}, FP32, eval+no_grad, CUDA sync")
    print(f"Latency: {lat_mean:.1f} ± {lat_std:.1f} ms/img")
    print(f"FPS: {fps_mean:.2f} ± {fps_std:.2f}")

    return (lat_mean, lat_std), (fps_mean, fps_std)


# ---------------------------
# ✅ THOP 统计辅助函数
# ---------------------------
def _conv2d_macs(batch, out_h, out_w, in_c, out_c, k=3, groups=1):
    # THOP 口径：MACs（不乘2）
    return batch * out_h * out_w * out_c * (in_c // groups) * k * k


def _count_wrapper_conv(m, x, y):
    """统计自定义Conv wrapper的FLOPs（双模式通用）"""
    inp = x[0]
    out = y
    if isinstance(out, (tuple, list)):
        out = out[0]
    if not hasattr(m, "conv") or not isinstance(m.conv, nn.Conv2d):
        return

    batch = inp.shape[0]
    out_h, out_w = out.shape[2], out.shape[3]
    in_c = m.conv.in_channels
    out_c = m.conv.out_channels
    k = m.conv.kernel_size[0]
    groups = m.conv.groups

    macs = _conv2d_macs(batch, out_h, out_w, in_c, out_c, k=k, groups=groups)
    macs_tensor = torch.tensor([macs], dtype=torch.float64, device=inp.device)

    if m.total_ops.device != inp.device:
        m.total_ops = m.total_ops.to(inp.device)
    m.total_ops += macs_tensor
    m._thop_called = True


def _count_deconv_deploy(m, x, y):
    """统计Deploy模式DEConv的FLOPs（权重融合→1次卷积）"""
    inp = x[0]
    out = y
    if isinstance(out, (tuple, list)):
        out = out[0]

    child_called = False
    for name in ["conv1_1", "conv1_2", "conv1_3", "conv1_4"]:
        child = getattr(m, name, None)
        if child is not None and getattr(child, "_thop_called", False):
            child_called = True
        if child is not None and hasattr(child, "_thop_called"):
            child._thop_called = False

    if child_called:
        return

    batch = inp.shape[0]
    out_h, out_w = out.shape[2], out.shape[3]
    in_c = inp.shape[1]
    out_c = out.shape[1]

    macs = _conv2d_macs(batch, out_h, out_w, in_c, out_c, k=3, groups=1)
    macs_tensor = torch.tensor([macs], dtype=torch.float64, device=inp.device)

    if m.total_ops.device != inp.device:
        m.total_ops = m.total_ops.to(inp.device)
    m.total_ops += macs_tensor


def _count_deconv_trainstruct(m, x, y):
    """统计TrainStruct模式DEConv的FLOPs（4个分支独立卷积）"""
    inp = x[0]
    out = y
    if isinstance(out, (tuple, list)):
        out = out[0]

    child_called = False
    for name in ["conv1_1", "conv1_2", "conv1_3", "conv1_4"]:
        child = getattr(m, name, None)
        if child is not None and getattr(child, "_thop_called", False):
            child_called = True
        if child is not None and hasattr(child, "_thop_called"):
            child._thop_called = False

    if child_called:
        return

    batch = inp.shape[0]
    out_h, out_w = out.shape[2], out.shape[3]
    in_c = inp.shape[1]
    out_c = out.shape[1]

    single_macs = _conv2d_macs(batch, out_h, out_w, in_c, out_c, k=3, groups=1)
    total_macs = single_macs * 4  # 4 个分支

    macs_tensor = torch.tensor([total_macs], dtype=torch.float64, device=inp.device)

    if m.total_ops.device != inp.device:
        m.total_ops = m.total_ops.to(inp.device)
    m.total_ops += macs_tensor


def build_thop_custom_ops(model: nn.Module, mode="deploy"):
    """构建THOP自定义统计规则（支持双模式切换）"""
    custom_ops = {}
    for mm in model.modules():
        if mm.__class__.__name__ == "DEConv":
            if mode == "deploy":
                custom_ops[type(mm)] = _count_deconv_deploy
            elif mode == "trainstruct":
                custom_ops[type(mm)] = _count_deconv_trainstruct

        if mm.__class__.__name__ in ["Conv2d_cd", "Conv2d_hd", "Conv2d_vd", "Conv2d_ad"]:
            custom_ops[type(mm)] = _count_wrapper_conv
    return custom_ops


# ---------------------------
# ✅ 模型复杂度统计（支持双模式）
# ---------------------------
def calculate_model_flops_and_params(opt, input_size=(3, 480, 640), pad_to=4, mode="deploy"):
    """
    mode:
      - "deploy": 统计 folded 口径（融合后的等效卷积）
      - "trainstruct": 统计 unfolded 口径（分支独立卷积）
    """
    temp_model = Backbone(opt).cuda()
    temp_model.eval()

    # 可选：加载权重
    if hasattr(opt, "pre_trained_model") and opt.pre_trained_model and os.path.isfile(opt.pre_trained_model):
        load_ckpt_safely(temp_model, opt.pre_trained_model)

    # 如果想在 deploy 模式下统计“真正折叠后的计算图”，这里可以尝试切到 deploy
    # （如果你的模型实现支持）
    if mode == "deploy":
        try_switch_to_deploy(temp_model)

    total_params = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
    params_million = total_params / 1e6

    print(f"\n=== 模型复杂度统计（{mode.upper()} 模式）===")
    print(f"模型参数量: {params_million:.4f} M ({total_params:,} )")

    dummy_input = torch.randn(1, *input_size, device="cuda", requires_grad=False)
    dummy_input = pad_img(dummy_input, pad_to)

    custom_ops = build_thop_custom_ops(temp_model, mode=mode)

    with torch.no_grad():
        flops, params = profile(
            temp_model,
            inputs=(dummy_input,),
            custom_ops=custom_ops if len(custom_ops) > 0 else None,
            verbose=False
        )

    flops_formatted, params_formatted = clever_format([flops, params], "%.4f")
    print(f"模型FLOPs: {flops_formatted}")
    print(f"参数量（thop验证）: {params_formatted}")

    del temp_model
    torch.cuda.empty_cache()

    return params_million, flops


# ---------------------------
# ✅ 主函数
# ---------------------------
if __name__ == "__main__":
    # cuDNN 自动选择最优算法（需 warm-up）
    torch.backends.cudnn.benchmark = True

    # ======================
    # 模式切换：deploy / trainstruct
    # ======================
    mode = "trainstruct"  # 改这里： "deploy" or "trainstruct"

    # 1) 统计 Params + FLOPs
    params_million, flops = calculate_model_flops_and_params(
        opt,
        input_size=INPUT_SIZE,
        pad_to=PAD_TO,
        mode=mode
    )

    # 2) 初始化模型用于测速
    network = Backbone(opt).cuda()
    network.eval()

    # 若测试 Deploy，尽量让测速也切换到 deploy（如果实现支持）
    if mode == "deploy":
        try_switch_to_deploy(network)

    # 3) 加载权重（测速建议加载，和论文一致；但没有也能测）
    if hasattr(opt, "pre_trained_model") and opt.pre_trained_model and os.path.isfile(opt.pre_trained_model):
        load_ckpt_safely(network, opt.pre_trained_model)
    else:
        print("⚠️ 未找到预训练权重，将使用随机初始化模型测速:", getattr(opt, "pre_trained_model", None))

    # 4) CUDA Event 测速（固定 warm-up=100、runs=500，并输出 mean±std）
    benchmark_latency_fps_cuda(
        network,
        input_size=INPUT_SIZE,
        pad_to=PAD_TO,
        warmup=WARMUP,
        runs=RUNS,
        trials=TRIALS
    )

    print("\n✅ 模型参数/性能统计完成！")
