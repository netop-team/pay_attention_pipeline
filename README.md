# Pay attention pipeline


This repository provides a pipeline for computing **Influence** and performing **Generation with GUIDE** to enhance transformer model explainability and performance. The pipeline leverages attention scores and embedding vectors to assess the importance of specific subsequences and applies various levels of instruction enhancement to improve model responses.

## Features

- **Generation with GUIDE**: Use guided instruction to generate more accurate and contextually relevant outputs from transformer models.
- **Influence Calculation**: Assess the impact of specific subsequences on the model's predictions using attention scores and embedding vectors.

# Motivation

Large Language Models (LLMs) are currently the state-of-the-art in most NLP tasks. Despite their success, pretrained LLMs sometimes struggle to accurately interpret diverse users' instructions and may generate outputs that do not align with human expectations. Additionally, LLMs can produce biased or hallucinated facts, which can limit their practical usefulness. 

Other work indicate that transformers are less likely to align with instructions as the context length grows. In such cases, rather than fulfilling the user's request, the model generates nonsensical text or repeats segments from the prompt.

A common solution to this problem is Supervised Fine Tuning (SFT) and Reinforcement Learning. However, these approaches are resource-intensive, time-consuming, and sensitive to specific data and tasks. Ideally, a more efficient approach would be one that, once implemented, does not require additional training.

In that sense, due to its low cost and broad accessibility, prompt engineering is widely used to align the outputs of LLMs with user preferences. However, this method does not always produce consistent results and can be very unstable.

We present GUIDE (**G**uided **U**nderstanding with **I**nstruction-**D**riven **E**nhancements): a systematic approach that allows users to emphasize instructions in their prompts.


# GUIDE 

GUIDE  is a novel and systematic approach that enables users to highlight critical instructions within the text input provided to an LLM. This pipeline implements GUIDE and enables users to influence the attention given to specific tokens by simply enclosing important text within tags like ```<!-> <-!>``` (as shown below). We propose to achieve this by simply adding a bias, denoted by $\Delta$, to the attention logits of the important tokens, i.e., $\bar{w}_{k,i}^{(\ell)} = w_{k,i}^{(\ell)} + \Delta,$ for all tokens indicated by the user, as shown by the attention matrices below, where each entry represents the impact of a past token (x-axis) on the ongoing token (y-axis).

![GUIDE](img/PayAttentionToWhatMatters-Workshop-extended.drawio.png)

Our results show that GUIDE substantially improves the accuracy of following certain instructions, outperforming natural prompting alternatives and Supervised Fine Tuning up to 1M tokens.


## Installation


To set up the environment for this project, follow these steps:

**Install python package**

   ```bash
   pip install pay-attention-pipeline
   ```


## Usage

Normally, one would load a pipeline using the Hugging Face pipeline as shown below:

```python
from transformers import pipeline

pipe = pipeline(
   "text-generation",
   model="your_model_name",
)

prompt = '''
   The Eiffel Tower, an iconic symbol of Paris and France, was completed in 1889 as the centerpiece of the Exposition Universelle, a world’s fair celebrating the centennial of the French Revolution...
   '''

out = pipe("Rewrite in French" + prompt, max_new_tokens = 100)
```

However, with this repository, you can use our custom ```PayAttentionPipeline``` to take advantage of the specialized features provided here: GUIDE and Influence.

If your prompt does not contain the tags `<?-> <-?>`, `<!-> <-!>`, `<!!-> <-!!>` or `<!!!-> <-!!!>`, our pipeline works exactly the same as HuggingFace's one

### Influence Calculation

The influence metric assesses the importance of a subsequence in the context of the model's predictions. Here’s how to compute it:

1. **Load the Pipeline**

   ```python
   from transformers import pipeline
   from transformers.pipelines import PIPELINE_REGISTRY
   from pay_attention_pipeline import PayAttentionPipeline

   pipe = pipeline(
       "pay-attention",
       model="mistralai/Mistral-7B-Instruct-v0.1",
   )

   PIPELINE_REGISTRY.check_task("pay-attention") # check if the pipeline is correctly loaded
   prompt = "Add you prompt here"
   ```

2. **Apply GUIDE Levels**

   Enhance the generation using various levels of instruction:

   ```python
   message_1 = [{'role': 'user', 'content': "<!-> Rewrite in French: <-!>" + prompt}]
   out_1 = pipe(message_1, max_new_tokens=100)

   message_2 = [{'role': 'user', 'content': "<!!-> Rewrite in French: <-!!>" + prompt}]
   out_2 = pipe(message_2, max_new_tokens=100)

   message_3 = [{'role': 'user', 'content': "<!!!-> Rewrite in French: <-!!!>" + prompt}]
   out_3 = pipe(message_3, max_new_tokens=100)
   ```

   Adjust the enhancement values as needed for your task.


3. **Customizing (Optional)**

   To experiment with other values of delta, set `delta_mid`:

   ```python
   dumb_pipe = pipeline(
       "pay-attention",
       model=base_model,
       tokenizer=tokenizer,
       model_kwargs=dict(cache_dir="/Data"),
       **dict(delta_mid=10)
   )
   message = [{'role': 'user', 'content': "<!!-> Rewrite in French:  <-!!>" + prompt}]
   out = dumb_pipe(message, max_new_tokens=100)
   ```

# Influence 

While GUIDE does not require additional training, it does necessitate the careful selection of how much to increase attention weights. In our study, we propose default values for certain tasks, but we also recognize the need to quantify these adjustments. To address this, we introduce a novel metric called *Influence*. This metric measures the importance of specific tokens in relation to instruction tokens within the text, and we use it to determine reasonable values for the increase in attention weights.

1. **Compute Influence**

   Wrap the text you want to compute the importance between `<?-> <-?>` tokens. The output will contain a dictionary of tensors, where each tensor represents the importance of the enclosured text through context length. 

   We provide two metrics you can use: Attention Rollout and Influence. Influence is computed by default

   
   If you want to compute the importance with a $\Delta \neq =0$, add the parameter `delta_influence` For example:

   ```python
   prompt = '''
   The Eiffel Tower, an iconic symbol of Paris and France, was completed in 1889 as the centerpiece of the Exposition Universelle, a world’s fair celebrating the centennial of the French Revolution...
   '''
   out1 = pipe("<?-> Rewrite in French <-?>" + prompt, max_new_tokens=100)
   out2 = pipe("<?-> Rewrite in French <-?>" + prompt, max_new_tokens=100, delta_influence = 1)
   out3 = pipe("<?-> Rewrite in French <-?>" + prompt, max_new_tokens=100, metric = 'attention_rollout')   
   out_caps = pipe("<?-> REWRITE IN FRENCH <-?>" + prompt, max_new_tokens = 100, )
   

   influence = out1['influence']
   influence_delta = out2['influence']
   rollout = out3['influence']
   influence_caps = out_caps['influence']
   ```

   You can visualize the influence of different layers as follows. We also provide [here](examples/influence.ipynb) one example of plotting using HoloViews.
<!-- 
   ```python
   import torch
   import torch.nn.functional as F
   import matplotlib.pyplot as plt

   def rolling_mean(x, window_size):
       # (Function implementation)

   layers_to_plot = [0, 15, 31]
   layers_to_axs_idx = {v: i for i, v in enumerate(layers_to_plot)}
   n_plots = len(layers_to_plot)
   fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * 5, 4))

   for layer_idx in layers_to_plot:
       plot_idx = layers_to_axs_idx[layer_idx]
       axes[plot_idx].plot(
           rolling_mean(torch.log(influence[layer_idx]), 10)[10:],
           label="Normal"
       )
       axes[plot_idx].plot(
           rolling_mean(torch.log(influence_caps[layer_idx]), 10)[10:],
           label="Uppercase"
       )

      axes[plot_idx].plot(
         rolling_mean(torch.log(influence_delta[layer_idx]), 10)[10:],
         label = r"$\Delta = 1$"
      )

      axes[plot_idx].plot(
         rolling_mean(torch.log(influence_delta[layer_idx]), 10)[10:],
         label = r"$\Delta = 1$"
      )
       axes[plot_idx].set_title(f"Layer {layer_idx+1}")
       axes[plot_idx].grid()
       axes[plot_idx].set_xlabel("context length")
       axes[plot_idx].set_ylabel("log influence")
       axes[plot_idx].legend()
   ``` -->

![Influence Plot](img/example_influence.png)

## Citation

If you use this for ressearch, please cite our paper.

```latex
@misc{silva2024payattention,
  title={Pay attention to what matters},
  author={Silva, Pedro Luiz and Ayed, Fadhel and de Domenico, Antonio and Maatouk, Ali},
  year={2024},
}
```


## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss potential improvements.
