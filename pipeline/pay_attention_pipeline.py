from transformers.pipelines import TextGenerationPipeline
from transformers.pipelines.text_generation import Chat, ReturnType
from src.GUIDE import GUIDEModel
from typing_extensions import override
import re
from enum import Enum
import torch
from src import Influence, AttentionRollout

class AttentionLevels(Enum):
    LEVEL_1= 1
    LEVEL_2= 2
    LEVEL_3= 3
    INFLUENCE= 4

class PayAttentionPipeline(TextGenerationPipeline):
    def __init__(
        self, 
        delta_mid : float = 1.,
        metric : str = "influence",
        num_layers : int = 32,
        *args, 
        **kwargs,
    ):
        self.num_layers = num_layers
        metric_options = ["influence", "attention_rollout"]
        assert metric in metric_options, f"metric must be one of {metric_options}"
        
        # ADD DOCSTRING
        super().__init__(*args, **kwargs)

        self.guide_model = GUIDEModel(
            self.model,
            self.tokenizer,
            should_save_params=False
        )
        
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

    def set_influence_model(self, metric : str):
        # TO BE CONTINUED
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
        self.instruction = instruction

    def __call__(self, text_inputs, metric : str = None, **kwargs):
        metric_options = ["influence", "attention_rollout", None]
        assert metric in metric_options, f"metric must be one of {metric_options}"

        self.set_influence_model(metric)

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

        if self._influence_tag[0] in prompt_text and self._influence_tag[1] in prompt_text:
            instruction, raw_instruction = PayAttentionPipeline._get_text_between(
                prompt_text,
                self._influence_tag[0],
                self._influence_tag[1]
            )

            self.mode = AttentionLevels.INFLUENCE
        
        elif self._enhance_attention_tag[AttentionLevels.LEVEL_1][0] in prompt_text and self._enhance_attention_tag[AttentionLevels.LEVEL_1][1] in prompt_text:
            instruction, raw_instruction = PayAttentionPipeline._get_text_between(
                prompt_text,
                self._enhance_attention_tag[AttentionLevels.LEVEL_1][0],
                self._enhance_attention_tag[AttentionLevels.LEVEL_1][1]
            )
            
            delta = self.levels[AttentionLevels.LEVEL_1]
            self.mode = AttentionLevels.LEVEL_1


        elif self._enhance_attention_tag[AttentionLevels.LEVEL_2][0] in prompt_text and self._enhance_attention_tag[AttentionLevels.LEVEL_2][1] in prompt_text:
            instruction, raw_instruction = PayAttentionPipeline._get_text_between(
                prompt_text,
                self._enhance_attention_tag[AttentionLevels.LEVEL_2][0],
                self._enhance_attention_tag[AttentionLevels.LEVEL_2][1]
            )
            
            delta = self.levels[AttentionLevels.LEVEL_2]
            self.mode = self.levels[AttentionLevels.LEVEL_2]
        
        elif self._enhance_attention_tag[AttentionLevels.LEVEL_3][0] in prompt_text and self._enhance_attention_tag[AttentionLevels.LEVEL_3][1] in prompt_text:
            instruction, raw_instruction = PayAttentionPipeline._get_text_between(
                prompt_text,
                self._enhance_attention_tag[AttentionLevels.LEVEL_3][0],
                self._enhance_attention_tag[AttentionLevels.LEVEL_3][1]
            )
            
            delta = self.levels[AttentionLevels.LEVEL_3]
            self.mode = self.levels[AttentionLevels.LEVEL_3]


        else:
            return super().preprocess(prompt_text, prefix, handle_long_generation, add_special_tokens, truncation, padding, max_length, **generate_kwargs)
        
        self.guide_model.set_delta_attention(delta)
        
        if isinstance(prompt_text, Chat):
            message = prompt_text

        else:
            message = [{"role": "user", "content": prompt_text}]

        template = self.tokenizer.apply_chat_template(
            message,
            tokenize= False
        )
         
        splits = template.split(instruction)
        print(instruction)
        print(splits)
        initial_prompt = splits[0]
        context = raw_instruction.join(splits[1:])

        prompt_text = initial_prompt + raw_instruction + context

        initial_tokens = self.tokenizer.encode(initial_prompt, return_tensors='pt', add_special_tokens = False)
        instruction_tokens = self.tokenizer.encode(raw_instruction, return_tensors='pt', add_special_tokens = False)
        context_tokens = self.tokenizer.encode(context, return_tensors='pt', add_special_tokens = False)

        start_idx = initial_tokens.size(1)
        end_idx = start_idx + instruction_tokens.size(1)
        
        tokens = torch.concat([
            initial_tokens.squeeze(), 
            instruction_tokens.squeeze(),
            context_tokens.squeeze()
        ]).unsqueeze(0)\
            .to(torch.int)

        # double checking
        instruction_words = self.tokenizer.decode(tokens.squeeze()[start_idx: end_idx])
        assert raw_instruction in instruction_words, "Error in tokenization. Not giving attention to correct tokens"

        self.guide_model.set_reference_tokens(start_idx, end_idx)
        self.guide_model.insert_hook()

        print(f"Inserting special attention on '{instruction_words}'")
        
        inputs = {
            "input_ids": tokens,
            "prompt_text" : prompt_text
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
        print(prompt)
        print(self.instruction)
        records = super().postprocess(model_outputs, return_type, clean_up_tokenization_spaces)

        if self.mode != AttentionLevels.INFLUENCE:
            self.guide_model.remove_hooks()
            self.guide_model.set_delta_attention(0)

        else:
            influence = self.influence_model(
                prompt,
                self.instruction,
                delta_attention= 0
            )

            records[0]["influence"]= influence

        return records

        

        
        


