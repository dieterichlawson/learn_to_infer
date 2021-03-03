import orchestration as orch
import argparse

parser = argparse.ArgumentParser(description='Run L2I commands.')

parser.add_argument('--init_tpus', action="store_true",
                    help='Start new TPUs for these commands')
parser.add_argument('--dry_run', action="store_true",
                    help='Do a dry run printing commands.')

args = parser.parse_args()

experiment_name = "lda_perp_exp"
command = "python3 learn_to_infer/run_lda.py"

job_args = {
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

commands = orch.make_commands(command, job_args)

if args.init_tpus:
  orch.initialize_tpus(experiment_name, len(commands), args.dry_run)

orch.run_commands(experiment_name, commands, args.dry_run)
