experiment_name = "lda_perp_exp"
command = "python3 learn_to_infer/run_lda.py"

hparams = {
  "model": "topic_word",
  "num_encoders": 2,
  "num_decoders": 2,
  "num_heads": 16,
  "key_dim": 32,
  "value_dim_per_head": 32,
  "embedding_dim": 64,
  "num_docs": [100, 250, 500, 1000],
  "num_topics": [5, 10, 20],
  "vocab_size": 1000,
  "doc_length": 50,
  "batch_size": 64,
  "eval_batch_size": 128,
  "num_steps": int(1e8),
  "lr": 1e-3,
  "summarize_every": 2500,
  "checkpoint_every": 2500,
  "logdir": "gs://l2i/" + experiment_name
}
