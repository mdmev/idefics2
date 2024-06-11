from transformers import AutoProcessor, BitsAndBytesConfig, Idefics2ForConditionalGeneration
from peft import LoraConfig
from transformers import BitsAndBytesConfig

class ModelLoader:
    @staticmethod
    def load_processor(model_name):
        return AutoProcessor.from_pretrained(
            model_name,
            do_image_splitting=True,
        )

    @staticmethod
    def load_model(model_name, use_qlora, use_lora, dtype, cache_dir):
        
        
        if use_qlora or use_lora:

            lora_config = LoraConfig(
                r=8,
                lora_alpha=8,
                lora_dropout=0.1,
                target_modules='.*(text_model|modality_projection|perceiver_resampler|vision_model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$',
                use_dora=False if use_qlora else True,
                init_lora_weights="gaussian"
            )
            if use_qlora:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype
                )
            model = Idefics2ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
                quantization_config=bnb_config if use_qlora else None,
                use_cache=False,
            )
            model.add_adapter(lora_config)
            model.enable_adapters()
        else:
            model = Idefics2ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
                use_cache=False,
                cache_dir=cache_dir,
            )
        return model
