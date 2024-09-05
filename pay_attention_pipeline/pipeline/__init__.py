from transformers.pipelines import pipeline, PIPELINE_REGISTRY

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from transformers.configuration_utils import PretrainedConfig
from transformers.image_processing_utils import BaseImageProcessor
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.feature_extraction_utils import PreTrainedFeatureExtractor
from transformers import AutoModelForCausalLM
from pay_attention_pipeline.pipeline.main_pipeline import PayAttentionPipeline

from transformers.pipelines.base import Pipeline

def is_torch_available():
    try:
        import torch
    except Exception as e:
        raise (e)
    
PIPELINE_REGISTRY.register_pipeline(
    "pay-attention",
    pipeline_class=PayAttentionPipeline,
    pt_model=AutoModelForCausalLM,
    default={"pt": ("mistralai/Mistral-7B-Instruct-v0.1", "1b62ab7")},
    type="text",  # current support type: text, audio, image, multimodal
)
