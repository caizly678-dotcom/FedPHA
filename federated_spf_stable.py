"""Stable, auditable runner for SPF-FedPHA.

This entry point avoids changing the legacy federated_main.py workflow. It
makes the temporal contract explicit: a client is evaluated with the frozen
server anchor that was used while training its private prompt in that round.
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import torch

# Import Dassl first so GL_SVDMSE is fully registered before patching it.
from Dassl.dassl.engine import build_trainer
from Dassl.dassl.optim import build_lr_scheduler, build_optimizer
from Dassl.dassl.utils import set_random_seed, setup_logger
from federated_main import print_args, setup_cfg
from utils.fed_utils import average_weights, count_parameters

# 必须放在 Dassl.engine 和 federated_main 之后，但在 build_trainer() 之前。
# 此时 GL_SVDMSE 已经完成注册，避免循环导入。
import trainers.spf_stable_patch  # noqa: F401


def build_parser():
    parser = argparse.ArgumentParser("Stable SPF-FedPHA runner")
    parser.add_argument("--trainer", type=str, default="GL_SVDMSE")
    parser.add_argument("--dataset", type=str, default="dtd")
    parser.add_argument("--backbone", type=str, default="ViT-B/16")
    parser.add_argument("--head", type=str, default="")
    parser.add_argument("--root", type=str, default="./data")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--transforms", type=str, nargs="+")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device_id", type=int, default=0)

    parser.add_argument("--num_users", type=int, default=10)
    parser.add_argument("--frac", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--iid", default=False)
    parser.add_argument("--partition", type=str, default="noniid-labeldir")
    parser.add_argument("--num_shots", type=int, default=2)
    parser.add_argument("--useall", default=False)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=1.0)

    parser.add_argument("--n_ctx", type=int, default=16)
    parser.add_argument("--num_prompt", type=int, default=2)
    parser.add_argument("--avg_prompt", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--ratio", type=float, default=0.8)
    parser.add_argument("--use_spf", action="store_true")
    parser.set_defaults(use_spf=True)
    parser.add_argument("--spf_energy", type=float, default=0.90)
    parser.add_argument("--spf_min_rank", type=int, default=1)
    parser.add_argument("--spf_max_rank", type=int, default=8)
    parser.add_argument("--spf_gamma_init", type=float, default=0.05)
    parser.add_argument("--spf_shared_lambda", type=float, default=0.0)
    parser.add_argument("--spf_local_lambda", type=float, default=0.5)
    parser.add_argument("--spf_global_lambda", type=float, default=1.0)

    parser.add_argument("--specify", default=False)
    parser.add_argument("--prompts_lens", nargs="+", type=int)
    parser.add_argument("--logdir", type=str, default="./logs/")
    parser.add_argument("--output_dir", type=str, default="output/..")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--load-epoch", type=int)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER)
    return parser


def prompt_learner(trainer):
    model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    return model.prompt_learner


def reset_prompt_optimizer(trainer):
    trainer.optim = build_optimizer(trainer.model.prompt_learner, trainer.cfg.OPTIM)
    trainer.sched = build_lr_scheduler(trainer.optim, trainer.cfg.OPTIM)
    trainer._optims["prompt_learner"] = trainer.optim
    trainer._scheds["prompt_learner"] = trainer.sched


def clone_cpu(tensor):
    return tensor.detach().cpu().clone()


def load_client_state(trainer, server_ctx, private_state, refresh_anchor):
    learner = prompt_learner(trainer)
    with torch.no_grad():
        learner.ctx_global.copy_(server_ctx.to(learner.ctx_global.device, learner.ctx_global.dtype))
        learner.ctx_local.copy_(private_state["ctx_local"].to(learner.ctx_local.device, learner.ctx_local.dtype))
        learner.spf_anchor_ctx.copy_(
            private_state["anchor_ctx"].to(learner.spf_anchor_ctx.device, learner.spf_anchor_ctx.dtype)
        )
        if refresh_anchor:
            learner.refresh_spf_anchor()


def save_private_state(trainer):
    learner = prompt_learner(trainer)
    return {
        "ctx_local": clone_cpu(learner.ctx_local),
        "anchor_ctx": clone_cpu(learner.spf_anchor_ctx),
    }


@torch.no_grad()
def evaluate_mode(trainer, client_id, mode):
    trainer.set_model_mode("eval")
    trainer.evaluator.reset()
    for batch in trainer.fed_test_loader_x_dict[client_id]:
        image, label = trainer.parse_batch_test(batch)
        logits = trainer.model(image, forward_mode=mode)
        trainer.evaluator.process(logits, label)
    result = trainer.evaluator.evaluate()
    trainer.evaluator.reset()
    return result


def summarize(results, sample_counts):
    accuracies = np.asarray([float(item["accuracy"]) for item in results], dtype=np.float64)
    counts = np.asarray(sample_counts, dtype=np.float64)
    return {
        "micro": float(np.average(accuracies, weights=counts)),
        "macro": float(accuracies.mean()),
        "worst": float(accuracies.min()),
        "std": float(accuracies.std()),
        "per_client": accuracies.tolist(),
    }


def append_row(path, row):
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main(args):
    if args.trainer != "GL_SVDMSE":
        raise ValueError("federated_spf_stable.py only supports --trainer GL_SVDMSE")

    cfg = setup_cfg(args)
    cfg.defrost()
    cfg.TRAINER.GL_SVDMSE.SPF_LOCAL_LAMBDA = args.spf_local_lambda
    cfg.TRAINER.GL_SVDMSE.SPF_GLOBAL_LAMBDA = args.spf_global_lambda
    cfg.TRAINER.GL_SVDMSE.SPF_SHARED_LAMBDA = args.spf_shared_lambda
    cfg.freeze()

    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)
    args.para_dir = setup_logger(cfg)
    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True
    print_args(args, cfg)

    trainer = build_trainer(args, cfg)
    trainer.fed_before_train()
    count_parameters(trainer.model, "prompt_learner")

    learner = prompt_learner(trainer)
    server_ctx = clone_cpu(learner.ctx_global)
    private_states = [
        {"ctx_local": clone_cpu(learner.ctx_local), "anchor_ctx": clone_cpu(learner.ctx_global)}
        for _ in range(cfg.DATASET.USERS)
    ]
    train_sizes = [len(trainer.fed_train_loader_x_dict[idx].dataset) for idx in range(cfg.DATASET.USERS)]
    test_sizes = [len(trainer.fed_test_loader_x_dict[idx].dataset) for idx in range(cfg.DATASET.USERS)]
    metrics_path = os.path.join(args.para_dir, "spf_stable_metrics.csv")
    history = []

    for round_id in range(cfg.OPTIM.ROUND):
        uploads = []
        round_start = time.time()
        for client_id in range(cfg.DATASET.USERS):
            # The anchor becomes the exact server prompt dispatched to this
            # client. It remains fixed for all of this client's local updates.
            load_client_state(trainer, server_ctx, private_states[client_id], refresh_anchor=True)
            reset_prompt_optimizer(trainer)
            trainer.train(idx=client_id, global_epoch=round_id, is_fed=True)
            private_states[client_id] = save_private_state(trainer)
            uploads.append(clone_cpu(prompt_learner(trainer).ctx_global))

        server_ctx = average_weights(uploads, list(range(cfg.DATASET.USERS)), train_sizes, islist=True)

        global_results, local_results, fused_results = [], [], []
        for client_id in range(cfg.DATASET.USERS):
            # Use fresh server parameters for strict GM, but use the saved
            # training-time anchor for the personalized fused model.
            load_client_state(trainer, server_ctx, private_states[client_id], refresh_anchor=False)
            global_results.append(evaluate_mode(trainer, client_id, "global_only"))
            local_results.append(evaluate_mode(trainer, client_id, "local_only"))
            fused_results.append(evaluate_mode(trainer, client_id, "fused"))

        gm = summarize(global_results, test_sizes)
        pm_local = summarize(local_results, test_sizes)
        pm_fused = summarize(fused_results, test_sizes)
        row = {
            "round": round_id,
            "gm_micro": gm["micro"],
            "gm_macro": gm["macro"],
            "pm_fused_micro": pm_fused["micro"],
            "pm_fused_macro": pm_fused["macro"],
            "pm_local_micro": pm_local["micro"],
            "pm_local_macro": pm_local["macro"],
            "fused_minus_local": pm_fused["micro"] - pm_local["micro"],
            "worst_client_fused": pm_fused["worst"],
            "client_std_fused": pm_fused["std"],
            "round_seconds": time.time() - round_start,
        }
        append_row(metrics_path, row)
        history.append(row)
        print(
            f"Round {round_id:03d} | GM={gm['micro']:.2f} | "
            f"PM-fused={pm_fused['micro']:.2f} | PM-local={pm_local['micro']:.2f} | "
            f"Fused-Local={row['fused_minus_local']:+.2f} | "
            f"Worst={pm_fused['worst']:.2f}"
        )

    summary = {
        "final": history[-1] if history else {},
        "best_gm_micro": max((row["gm_micro"] for row in history), default=float("nan")),
        "best_pm_fused_micro": max((row["pm_fused_micro"] for row in history), default=float("nan")),
        "best_pm_local_micro": max((row["pm_local_micro"] for row in history), default=float("nan")),
        "config": {
            "gamma": args.spf_gamma_init,
            "global_lambda": args.spf_global_lambda,
            "local_lambda": args.spf_local_lambda,
            "shared_lambda": args.spf_shared_lambda,
        },
    }
    with open(os.path.join(args.para_dir, "spf_stable_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    trainer.fed_after_train()


if __name__ == "__main__":
    main(build_parser().parse_args())
