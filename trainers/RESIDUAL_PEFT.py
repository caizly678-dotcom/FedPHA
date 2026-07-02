import os.path as osp
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from Dassl.dassl.engine.trainer import TrainerX
from Dassl.dassl.metrics import compute_accuracy
from Dassl.dassl.utils import load_pretrained_weights, load_checkpoint
from Dassl.dassl.optim import build_optimizer, build_lr_scheduler
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {"trainer": 'GL_SVDMSE',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}

    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class LowRankAsymmetricCrossAttention(nn.Module):
    """
    Local context anchor -> detached global context tokens.
    q: [N, 1, D]
    k/v: [N, n_ctx, D]
    """
    def __init__(self, dim, rank=32, dtype=torch.float32):
        super().__init__()

        self.scale = rank ** -0.5

        self.q_proj = nn.Linear(dim, rank, bias=False)
        self.k_proj = nn.Linear(dim, rank, bias=False)
        self.v_proj = nn.Linear(dim, rank, bias=False)
        self.o_proj = nn.Linear(rank, dim, bias=False)

        # 🌟 核心修改 1：打破零初始化带来的梯度死锁，采用微光注入
        nn.init.normal_(self.o_proj.weight, std=1e-4)

        self.to(dtype=dtype)

    def forward(self, q, k, v):
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = attn @ v
        return self.o_proj(out)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.RESIDUAL_PEFT.N_CTX
        ctx_init = cfg.TRAINER.RESIDUAL_PEFT.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.N = cfg.TRAINER.RESIDUAL_PEFT.N
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
            ctx_global = ctx_vectors.unsqueeze(0).repeat(self.N, 1, 1)
            ctx_local = ctx_vectors.unsqueeze(0).repeat(self.N, 1, 1)

        else:
            if cfg.TRAINER.RESIDUAL_PEFT.CSC:
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                ctx_global = torch.empty(self.N, n_ctx, ctx_dim, dtype=dtype)
                ctx_local = torch.empty(self.N, n_ctx, ctx_dim, dtype=dtype) 
            
            nn.init.normal_(ctx_global, std=0.02)
            nn.init.normal_(ctx_local, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)    

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx_global = nn.Parameter(ctx_global)
        self.ctx_local = nn.Parameter(ctx_local)
        
        self.lraca = LowRankAsymmetricCrossAttention(
            dim=ctx_dim,
            rank=cfg.TRAINER.RESIDUAL_PEFT.LRACA_RANK,
            dtype=dtype
        )

        # 🌟 核心修改 2：已清理冗余的 alpha_logit 和 residual_tau
        
        classnames = [name.replace("_", " ") for name in classnames]   
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])   
        tokenized_prompts = tokenized_prompts.repeat(self.N, 1) 

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) 

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.RESIDUAL_PEFT.CLASS_TOKEN_POSITION

    def build_prompts(self, ctx):
        ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
        ctx = ctx.permute(1, 0, 2, 3).contiguous()
        ctx = ctx.view(self.N * self.n_cls, self.n_ctx, ctx.shape[-1])

        prompts = torch.cat(
            [self.token_prefix, ctx, self.token_suffix],
            dim=1
        )
        return prompts

    def forward(self):
        if self.class_token_position != "end":
            raise NotImplementedError(
                "RESIDUAL_PEFT currently supports CLASS_TOKEN_POSITION == 'end'"
            )

        # Global Prompt：独立生成
        ctx_global = self.ctx_global

        # Local Prompt：仅第一个 Context Token 查询 Global Context
        local_anchor = self.ctx_local[:, :1, :]
        global_context = ctx_global.detach()

        cross_anchor = self.lraca(
            q=local_anchor,
            k=global_context,
            v=global_context
        )

        # 🌟 核心修改 2：完全在 Prompt 空间完成残差知识叠加！
        ctx_local_cond = torch.cat(
            [
                self.ctx_local[:, :1, :] + cross_anchor,
                self.ctx_local[:, 1:, :]
            ],
            dim=1
        )

        prompts_global = self.build_prompts(ctx_global)
        prompts_local = self.build_prompts(ctx_local_cond)

        return prompts_global, prompts_local


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.n_cls = len(classnames)
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.N = cfg.TRAINER.RESIDUAL_PEFT.N

    def train(self, mode=True):
        super().train(mode)
        # 冻结 encoder 始终 eval
        self.image_encoder.eval()
        self.text_encoder.eval()
        return self

    def forward(self, image, idx=None, mode="both"):
        tokenized_prompts = self.tokenized_prompts

        # 接收融合好全局知识的 Local Prompts
        prompts_global, prompts_local = self.prompt_learner()

        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True
            )

        # TextEncoder 不可以 no_grad，梯度要流回 Prompt
        text_global = self.text_encoder(prompts_global, tokenized_prompts)
        text_global = text_global / text_global.norm(dim=-1, keepdim=True)

        text_local = self.text_encoder(prompts_local, tokenized_prompts)
        text_local = text_local / text_local.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()

        global_logits = logit_scale * image_features @ text_global.t()
        
        # 🌟 核心修改 2：拆除 Logit 层面的 tanh 和 alpha 枷锁，彻底释放爆发力！
        personalized_logits = logit_scale * image_features @ text_local.t()

        if mode == "global":
            return global_logits
        if mode == "personalized":
            return personalized_logits
        if mode == "both":
            return global_logits, personalized_logits, {}


# @TRAINER_REGISTRY.register()
class RESIDUAL_PEFT(TrainerX):
    """
    It is based on CoOp.
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.RESIDUAL_PEFT.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.RESIDUAL_PEFT.PREC == "fp32" or cfg.TRAINER.RESIDUAL_PEFT.PREC == "amp":
            clip_model.float()   

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        if cfg.DATASET.NAME== "ImageNet":
            self.device =  torch.device("cuda:0")
            device1 = torch.device("cuda")
            self.model.to(self.device)
            self.model.text_encoder.to(device1)
            self.model.text_encoder=nn.DataParallel(self.model.text_encoder)
        else:
            self.model.to(self.device)
        
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.RESIDUAL_PEFT.PREC == "amp" else None

    def forward_backward(self, batch_idx, batch, **kwargs):
        image, label = self.parse_batch_train(batch)

        global_logits, personalized_logits, aux = self.model(
            image, mode="both"
        )

        loss_global_ce = F.cross_entropy(global_logits, label)
        loss_personal = F.cross_entropy(personalized_logits, label)

        # =========================================================
        # Debug only: 梯度隔离安全检查 (保留原逻辑以保证学术严谨)
        # =========================================================
        if not getattr(self, "_gradient_isolation_checked", False):
            self.optim.zero_grad(set_to_none=True)
            loss_personal.backward(retain_graph=True)
            global_grad = self.model.prompt_learner.ctx_global.grad
            global_grad_max = (0.0 if global_grad is None else global_grad.detach().abs().max().item())

            print(f"[Gradient Isolation Check] max |dL_personal / dctx_global| = {global_grad_max:.3e}")
            assert global_grad_max < 1e-10, "Gradient leakage detected!"

            self.optim.zero_grad(set_to_none=True)
            loss_global_ce.backward(retain_graph=True)
            private_grad_max = 0.0
            for name, param in self.model.prompt_learner.named_parameters():
                if name == "ctx_local" or name.startswith("lraca."):
                    if param.grad is not None:
                        private_grad_max = max(private_grad_max, param.grad.detach().abs().max().item())

            print(f"[Gradient Isolation Check] max |dL_global / dprivate| = {private_grad_max:.3e}")
            assert private_grad_max < 1e-10, "Gradient leakage detected!"

            self.optim.zero_grad(set_to_none=True)
            self._gradient_isolation_checked = True

        # =========================================================
        # 正常训练
        # =========================================================
        # 🌟 核心修改 3：引入 L2 范数正则化，防止脱离本地约束的 Global 分支在强 Non-IID 下崩溃
        reg_global = 0.05 * torch.norm(global_logits, p=2).mean()
        loss_global = loss_global_ce + reg_global

        lambda_p = self.cfg.TRAINER.RESIDUAL_PEFT.LAMBDA_P
        loss = loss_global + lambda_p * loss_personal

        self.model_backward_and_update(loss)

        return {
            "loss": loss.item(),
            "loss_global": loss_global.item(),
            "loss_personal": loss_personal.item(),
            "global_acc": compute_accuracy(global_logits, label)[0].item(),
            "personal_acc": compute_accuracy(personalized_logits, label)[0].item(),
        }
    
    def set_eval_mode(self, mode):
        assert mode in ["global", "personalized"]
        self.eval_mode = mode

    def model_inference(self, input, idx):
        self.model.eval()
        return self.model(
            input,
            idx=idx,
            mode=self.eval_mode
        )

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return
        names = self.get_model_names()
        model_file = "model-best.pth.tar"
        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]
            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)