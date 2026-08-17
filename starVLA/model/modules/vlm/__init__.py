def get_vlm_model(config):

    vlm_name = config.framework.qwenvl.base_vlm
    vlm_name_lower = vlm_name.lower()
    configured_model_type = str(
        config.framework.qwenvl.get("model_type", "")
    ).lower().replace("-", "_")

    if (
        configured_model_type in {"qwen2_5", "qwen2_5_vl"}
        or "qwen2.5-vl" in vlm_name_lower
        or "nora" in vlm_name_lower  # temp for some ckpt
    ):
        from .QWen2_5 import _QWen_VL_Interface
        return _QWen_VL_Interface(config)
    elif (
        configured_model_type in {"qwen3", "qwen3_vl"}
        or "qwen3-vl" in vlm_name_lower
    ):
        from .QWen3 import _QWen3_VL_Interface
        return _QWen3_VL_Interface(config)
    elif (
        configured_model_type in {"qwen3_5", "qwen3.5"}
        or "qwen3.5" in vlm_name_lower
    ):
        from .QWen3_5 import _QWen3_5_VL_Interface
        return _QWen3_5_VL_Interface(config)
    elif "florence" in vlm_name_lower: # temp for some ckpt
        from .Florence2 import _Florence_Interface
        return _Florence_Interface(config)
    elif "cosmos-reason2" in vlm_name_lower:
        from .CosmosReason2 import _CosmosReason2_Interface
        return _CosmosReason2_Interface(config)
    else:
        raise NotImplementedError(f"VLM model {vlm_name} not implemented")
