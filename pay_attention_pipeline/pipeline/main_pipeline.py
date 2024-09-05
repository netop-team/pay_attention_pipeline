from transformers.pipelines import TextGenerationPipeline
from transformers.pipelines.text_generation import Chat, ReturnType
from ..src.GUIDE import GUIDEModel
from typing_extensions import override
import re
from enum import Enum
import torch
from ..src.metrics import Influence, AttentionRollout

class AttentionLevels(Enum):
    LEVEL_1= 1
    LEVEL_2= 2
    LEVEL_3= 3
    INFLUENCE= 4

class PayAttentionPipeline(TextGenerationPipeline):
    env = 'local'
    """
    A custom text generation pipeline that adds specialized attention mechanisms to generated text. 
    It supports different attention metrics such as influence and attention rollout, 
    and allows for dynamic adjustment of attention levels during text generation.

    Args:
        metric (str): The metric to use for attention. Options are "influence" or "attention_rollout". Default is "influence".
        num_layers (int): The number of layers in the model for which the attention mechanism will be applied. Default is 32.
        *args: Additional positional arguments for the parent class.
        **kwargs: Additional keyword arguments for the parent class.
    """
    def __init__(
        self, 
        metric : str = "influence",
        *args, 
        **kwargs,
    ):  
        
        if 'delta_mid' in kwargs:
            delta_mid = kwargs['delta_mid']
            kwargs.pop('delta_mid')
        else:
            delta_mid = 1.
        
        metric_options = ["influence", "attention_rollout"]
        assert metric in metric_options, f"metric must be one of {metric_options}"
        
        super().__init__(*args, **kwargs)
        self.num_layers = len(self.model.model.layers)

        self.guide_model = GUIDEModel(
            self.model,
            self.tokenizer,
            should_save_params=False
        )

        self.delta_influence =0 
        
        # add influence model
        self.set_influence_model(metric)

        self._influence_tag = ['<?->', '<-?>']
        self._enhance_attention_tag = {
            AttentionLevels.LEVEL_1: ["<!->", "<-!>"],
            AttentionLevels.LEVEL_2: ["<!!->", "<-!!>"],
            AttentionLevels.LEVEL_3: ["<!!!->", "<-!!!>"]
        }

        self.levels = {
            AttentionLevels.LEVEL_1: delta_mid/2,
            AttentionLevels.LEVEL_2: delta_mid,
            AttentionLevels.LEVEL_3: 2*delta_mid
        }

        self.mode : AttentionLevels = None
        self.instruction : str = None

    def remove_hooks(self):
        self.guide_model.remove_hooks()

    def set_influence_model(self, metric : str):
        """
        Sets the influence model based on the provided metric.

        Args:
            metric (str): The metric to use for the influence model. 
                          Must be either "influence" or "attention_rollout".
        """
        if metric == "influence":
            self.influence_model = Influence(
                self.model,
                self.tokenizer,
                self.num_layers 
            )

        elif metric == "attention_rollout":
            self.influence_model = AttentionRollout(
                self.model,
                self.tokenizer,
                self.num_layers
            )

    @staticmethod
    def _get_text_between(
        text : str, 
        start_word : str, 
        end_word: str
    ):
        """
        Extracts the text between two specified words within a string.

        Args:
            text (str): The text from which to extract.
            start_word (str): The word marking the start of the extraction.
            end_word (str): The word marking the end of the extraction.

        Returns:
            tuple: A tuple containing the full instruction (with tags) and the raw instruction (without tags).
        """
        # Escape tokens to handle special regex characters
        start_word_esp = re.escape(start_word)
        end_word_esp = re.escape(end_word)
        
        # Create the regex pattern
        pattern = f'{start_word_esp}(.*?){end_word_esp}'
        
        # Find all matches
        matches = re.findall(pattern, text)
        
    
        raw_instruction = matches[0].strip() if matches else None
        instruction = start_word+ matches[0] +end_word


        return instruction, raw_instruction 
    
    def set_instruction(self, instruction : str):
        """
        Sets the instruction that will be used for guiding the attention mechanism.

        Args:
            instruction (str): The instruction to be set.
        """
        self.instruction = instruction

    def __call__(self, text_inputs, metric : str = 'influence', delta_influence = 0,**kwargs):
        
        metric_options = ["influence", "attention_rollout", None]
        assert metric in metric_options, f"metric must be one of {metric_options}"

        
        self.set_influence_model(metric)
        self.delta_influence = delta_influence

        return super().__call__(text_inputs, **kwargs)
    
    @override
    def preprocess(
        self, 
        prompt_text, 
        prefix="", 
        handle_long_generation=None, 
        add_special_tokens=False, 
        truncation=None, 
        padding=False, 
        max_length=None, 
        **generate_kwargs
    ):
        instruction = None
        delta = 0
        
        if isinstance(prompt_text, Chat):
            messages = prompt_text.messages[0]['content']

        elif isinstance(prompt_text, str):
            messages = prompt_text

        if self._influence_tag[0] in messages and self._influence_tag[1] in messages:
            instruction, raw_instruction = PayAttentionPipeline._get_text_between(
                messages,
                self._influence_tag[0],
                self._influence_tag[1]
            )

            self.mode = AttentionLevels.INFLUENCE
        
        elif self._enhance_attention_tag[AttentionLevels.LEVEL_1][0] in messages and self._enhance_attention_tag[AttentionLevels.LEVEL_1][1] in messages:
            instruction, raw_instruction = PayAttentionPipeline._get_text_between(
                messages,
                self._enhance_attention_tag[AttentionLevels.LEVEL_1][0],
                self._enhance_attention_tag[AttentionLevels.LEVEL_1][1]
            )
            
            delta = self.levels[AttentionLevels.LEVEL_1]
            self.mode = AttentionLevels.LEVEL_1
            print(f"Applying level 1 enhancement to the prompt. Delta = {delta}")


        elif self._enhance_attention_tag[AttentionLevels.LEVEL_2][0] in messages and self._enhance_attention_tag[AttentionLevels.LEVEL_2][1] in messages:
            instruction, raw_instruction = PayAttentionPipeline._get_text_between(
                messages,
                self._enhance_attention_tag[AttentionLevels.LEVEL_2][0],
                self._enhance_attention_tag[AttentionLevels.LEVEL_2][1]
            )
            
            delta = self.levels[AttentionLevels.LEVEL_2]
            self.mode = self.levels[AttentionLevels.LEVEL_2]
            print(f"Applying level 2 enhancement to the prompt. Delta = {delta}")
        
        elif self._enhance_attention_tag[AttentionLevels.LEVEL_3][0] in messages and self._enhance_attention_tag[AttentionLevels.LEVEL_3][1] in messages:
            instruction, raw_instruction = PayAttentionPipeline._get_text_between(
                messages,
                self._enhance_attention_tag[AttentionLevels.LEVEL_3][0],
                self._enhance_attention_tag[AttentionLevels.LEVEL_3][1]
            )
            
            delta = self.levels[AttentionLevels.LEVEL_3]
            self.mode = self.levels[AttentionLevels.LEVEL_3]
            print(f"Applying level 3 enhancement to the prompt. Delta = {delta}")


        else:

            inputs = super().preprocess(prompt_text, prefix, handle_long_generation, add_special_tokens, truncation, padding, max_length, **generate_kwargs)
            return inputs
        
        self.guide_model.set_delta_attention(delta)
        
        if isinstance(prompt_text, Chat):
            message = prompt_text

        else:
            message = [{"role": "user", "content": prompt_text}]

        template = self.tokenizer.apply_chat_template(
            message,
            tokenize= False,
            add_generation_prompt=True
        )
         
        splits = template.split(instruction)
        initial_prompt = splits[0]
        context = raw_instruction.join(splits[1:])

        prompt_text = initial_prompt + raw_instruction + context

        initial_tokens = self.tokenizer.encode(initial_prompt, return_tensors='pt', add_special_tokens = False)
        instruction_tokens = self.tokenizer.encode(raw_instruction, return_tensors='pt', add_special_tokens = False)
        context_tokens = self.tokenizer.encode(context, return_tensors='pt', add_special_tokens = False)

        start_idx = initial_tokens.size(1)
        end_idx = start_idx + instruction_tokens.size(1)
        
        
        tokens = torch.concat([
            initial_tokens.squeeze(dim = 0), 
            instruction_tokens.squeeze(dim = 0),
            context_tokens.squeeze(dim=0)
        ]).unsqueeze(0)\
            .to(torch.int)

        # double checking
        instruction_words = self.tokenizer.decode(tokens.squeeze()[start_idx: end_idx])
        assert raw_instruction in instruction_words, "Error in tokenization. Not giving attention to correct tokens"

        self.guide_model.set_reference_tokens(start_idx, end_idx)
        self.influence_model.set_reference_tokens(start_idx, end_idx)

        if self.mode == AttentionLevels.INFLUENCE:
            self.guide_model.insert_hook()
            print(f"Computing influence on '{instruction_words}'. Delta = {self.delta_influence}")
        else:
            self.guide_model.insert_pre_hook()
        
        
        inputs = {
            "input_ids": tokens,
            "prompt_text" : message
        }

        self.set_instruction(raw_instruction)
        
        return inputs
    
    @override
    def _forward(self, model_inputs, **generate_kwargs):

        input_ids = model_inputs['input_ids']
        prompt_text = model_inputs['prompt_text']

        if input_ids.shape[1] == 0:
            input_ids = None
            attention_mask = None
            in_b = 1
        else:
            in_b = input_ids.shape[0]

        generated_sequence = self.guide_model.generate(
            model_inputs['input_ids'],
            **generate_kwargs
        )
        out_b = generated_sequence.shape[0]

        if self.framework == "pt":
            generated_sequence = generated_sequence.reshape(in_b, out_b // in_b, *generated_sequence.shape[1:])

        return {"generated_sequence": generated_sequence, "input_ids": input_ids, "prompt_text": prompt_text}
    
    @override
    def postprocess(
        self, 
        model_outputs, 
        return_type=ReturnType.FULL_TEXT, 
        clean_up_tokenization_spaces=True
    ):  
        prompt = model_outputs['prompt_text']
        tokens = model_outputs['input_ids']
        records = super().postprocess(model_outputs, return_type, clean_up_tokenization_spaces)
        records[0]['tokens'] = tokens

        if self.mode != AttentionLevels.INFLUENCE:
            self.guide_model.remove_hooks()
            self.guide_model.set_delta_attention(0)

        else:
            influence = self.influence_model(
                tokens,
                delta_attention= self.delta_influence
            )

            records[0]["influence"]= influence

        return records

        

        
        


