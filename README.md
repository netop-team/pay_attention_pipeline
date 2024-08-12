## Pay attention pipeline


This repository provides a pipeline for computing **Influence** and performing **Generation with GUIDE** to enhance transformer model explainability and performance. The pipeline leverages attention scores and embedding vectors to assess the importance of specific subsequences and applies various levels of instruction enhancement to improve model responses.

## Features

- **Influence Calculation**: Assess the impact of specific subsequences on the model's predictions using attention scores and embedding vectors.
- **Generation with GUIDE**: Use guided instruction to generate more accurate and contextually relevant outputs from transformer models.

## Installation

To set up the environment for this project, follow these steps:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/netop-team/pay_attention_pipeline.git
   cd pay_attention_pipeline
   ```

2. **Create and Activate a Virtual Environment**

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

3. **Install Required Packages**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Influence Calculation

The influence metric assesses the importance of a subsequence in the context of the model's predictions. Here’s how to compute it:

1. **Load the Pipeline**

   ```python
   from transformers import pipeline

   pipe = pipeline(
       "pay-attention",
       model="mistralai/Mistral-7B-Instruct-v0.1",
   )
   ```

2. **Compute Influence**

   Add the tag `<?-> <-?>` around the text you want to compute the influence for. For example:

   ```python
   prompt = '''
   The Eiffel Tower, an iconic symbol of Paris and France, was completed in 1889 as the centerpiece of the Exposition Universelle, a world’s fair celebrating the centennial of the French Revolution...
   '''
   out1 = pipe("<?-> Rewrite in French <-?>" + prompt, max_new_tokens=100)
   out2 = pipe("<?-> REWRITE IN FRENCH <-?>" + prompt, max_new_tokens=100)

   influence = out1['influence']
   influence_caps = out2['influence']
   ```

   You can visualize the influence of different layers as follows:

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
       axes[plot_idx].set_title(f"Layer {layer_idx+1}")
       axes[plot_idx].grid()
       axes[plot_idx].set_xlabel("context length")
       axes[plot_idx].set_ylabel("log influence")
       axes[plot_idx].legend()
   ```

![Influence Plot](img/example_influence.png)

### Generation with GUIDE

GUIDE (Guided Understanding with Instruction-Driven Enhancements) improves the model's response based on instruction levels. Here’s how to use it:

1. **Load the Generative Model**

   ```python
   from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

   pipe = pipeline(
       "pay-attention",
       model="mistralai/Mistral-7B-Instruct-v0.1",
   )
   ```

2. **Apply GUIDE Levels**

   Enhance the generation using various levels of instruction:

   ```python
   message_1 = [{'role': 'user', 'content': "<!-> Summarize in French <-!>" + prompt}]
   out_1 = pipe(message_1, max_new_tokens=100)

   message_2 = [{'role': 'user', 'content': "<!!-> Summarize in French <-!!>" + prompt}]
   out_2 = pipe(message_2, max_new_tokens=100)

   message_3 = [{'role': 'user', 'content': "<!!!-> Summarize in French <-!!!>" + prompt}]
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
   message = [{'role': 'user', 'content': "<!!-> Summarize in French <-!!>" + prompt}]
   out = dumb_pipe(message, max_new_tokens=100)
   ```

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss potential improvements.
