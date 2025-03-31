---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:178
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/xlm-r-distilroberta-base-paraphrase-v1
widget:
- source_sentence: Se eu comer [ALIMENTO], vou ultrapassar [META_KCAL]?
  sentences:
  - You've already consumed [KCAL_CONSUMIDAS] kcal out of [META_KCAL]. A [ALIMENTO]
    like this usually has around [CALORIAS] kcal, [PROTEINAS] g of protein, [LIPIDOS]
    g of fat, and [HIDRATOS] g of carbs.
  - O [ALIMENTO] com [CALORIAS] kcal e [PROTEINAS] g de proteína pode servir se queres
    algo com poucas calorias e boa proteína.
  - Em vez de [ALIMENTO], podes escolher algo com menos de [CALORIAS] kcal para não
    exceder [META_KCAL].
- source_sentence: Quero uma sugestão para o jantar, pois já atingi [KCAL_CONSUMIDAS].
  sentences:
  - Se [ALIMENTO] tiver [CALORIAS] e for superior a 150 kcal, vais ultrapassar [META_KCAL].
  - É uma opção viável, mas se as [CALORIAS] kcal de [ALIMENTO] te fazem passar [META_KCAL],
    escolhe algo mais leve.
  - Se estás em [KCAL_CONSUMIDAS], ao acrescentar [ALIMENTO] de [CALORIAS] kcal, somas
    tudo ao total.
- source_sentence: How can I stay within [META_KCAL] if I want [ALIMENTO] for lunch?
  sentences:
  - '[ALIMENTO] has about [CALORIAS] kcal, [PROTEINAS] g of protein, [LIPIDOS] g of
    fat, and [HIDRATOS] g of carbs.'
  - Um [ALIMENTO] médio pode acrescentar [CALORIAS] kcal ao teu total. Se estás em
    [KCAL_CONSUMIDAS], faz as contas.
  - To keep [META_KCAL], if [ALIMENTO] is [CALORIAS] kcal, watch your fat and carbs
    in other meals.
- source_sentence: Comi [ALIMENTO] e estou na dúvida se posso comer também [ALIMENTO].
  sentences:
  - Para um snack até ao jantar, [ALIMENTO] costuma ficar em [CALORIAS] kcal, [PROTEINAS]
    g de proteínas, [LIPIDOS] g de lípidos e [HIDRATOS] g de hidratos, cabendo em
    [META_KCAL].
  - Se já comeste [ALIMENTO], e queres [ALIMENTO], soma as calorias para não passar
    [META_KCAL].
  - If you added [ALIMENTO] plus juice, it might total [CALORIAS] kcal. Check if you're
    still under [META_KCAL].
- source_sentence: Já ingeri [KCAL_CONSUMIDAS] kcal hoje. Quanto devo ajustar se quiser
    comer [ALIMENTO]?
  sentences:
  - Um [ALIMENTO] para lanche pode rondar [CALORIAS] kcal, [PROTEINAS] g de proteínas,
    [LIPIDOS] g de lípidos e [HIDRATOS] g de hidratos.
  - The quickest way is to assume ~[PROTEINAS] g protein and [LIPIDOS] g fat for [CALORIAS]
    kcal of [ALIMENTO].
  - Tens [KCAL_CONSUMIDAS] kcal até agora. Um [ALIMENTO] adiciona por volta de [CALORIAS]
    kcal, [PROTEINAS] g de proteínas, [LIPIDOS] g de lípidos e [HIDRATOS] g de hidratos.
    Ajusta conforme necessário.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/xlm-r-distilroberta-base-paraphrase-v1

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/xlm-r-distilroberta-base-paraphrase-v1](https://huggingface.co/sentence-transformers/xlm-r-distilroberta-base-paraphrase-v1). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/xlm-r-distilroberta-base-paraphrase-v1](https://huggingface.co/sentence-transformers/xlm-r-distilroberta-base-paraphrase-v1) <!-- at revision da45da457830903b91ec80da60611e6ba5846008 -->
- **Maximum Sequence Length:** 128 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 128, 'do_lower_case': False}) with Transformer model: XLMRobertaModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Já ingeri [KCAL_CONSUMIDAS] kcal hoje. Quanto devo ajustar se quiser comer [ALIMENTO]?',
    'Tens [KCAL_CONSUMIDAS] kcal até agora. Um [ALIMENTO] adiciona por volta de [CALORIAS] kcal, [PROTEINAS] g de proteínas, [LIPIDOS] g de lípidos e [HIDRATOS] g de hidratos. Ajusta conforme necessário.',
    'Um [ALIMENTO] para lanche pode rondar [CALORIAS] kcal, [PROTEINAS] g de proteínas, [LIPIDOS] g de lípidos e [HIDRATOS] g de hidratos.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 178 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 178 samples:
  |         | sentence_0                                                                         | sentence_1                                                                         |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             |
  | details | <ul><li>min: 12 tokens</li><li>mean: 24.49 tokens</li><li>max: 38 tokens</li></ul> | <ul><li>min: 23 tokens</li><li>mean: 41.55 tokens</li><li>max: 77 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                         | sentence_1                                                                                                                                                           |
  |:---------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>If I want to stay under [META_KCAL] kcal, can I have [ALIMENTO] now?</code>                  | <code>[ALIMENTO] is about [CALORIAS] kcal, [PROTEINAS] g of protein, [LIPIDOS] g of fat, and [HIDRATOS] g of carbs. If that fits your [META_KCAL], go for it.</code> |
  | <code>Se o [ALIMENTO] tem cerca de 400 kcal, posso comer metade para ficar nos [META_KCAL]?</code> | <code>Se o [ALIMENTO] tiver ~[HIDRATOS] g de hidratos, reduz os hidratos noutras refeições para equilibrar.</code>                                                   |
  | <code>Estou a tentar decidir entre [ALIMENTO] e [ALIMENTO]. Qual tem menos calorias?</code>        | <code>O [ALIMENTO] pode ter [PROTEINAS] g de proteína por 100 g, a cerca de [CALORIAS] kcal nessa quantidade.</code>                                                 |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 4
- `disable_tqdm`: True
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 4
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: True
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch | Step |
|:-----:|:----:|
| 1.0   | 12   |
| 2.0   | 24   |
| 3.0   | 36   |
| 4.0   | 48   |


### Framework Versions
- Python: 3.12.9
- Sentence Transformers: 4.0.1
- Transformers: 4.49.0
- PyTorch: 2.6.0+cu126
- Accelerate: 1.5.2
- Datasets: 3.3.2
- Tokenizers: 0.21.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->