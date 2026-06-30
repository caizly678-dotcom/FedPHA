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
        # loading JIT archive
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

        # 初始时不注入 Global 信息
        nn.init.zeros_(self.o_proj.weight)

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
        n_ctx = cfg.TRAINER.GL_SVDMSE.N_CTX
        ctx_init = cfg.TRAINER.GL_SVDMSE.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.N = cfg.TRAINER.GL_SVDMSE.N
        self.ratio = cfg.TRAINER.GL_SVDMSE.ratio
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
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
            # random initialization
            if cfg.TRAINER.GL_SVDMSE.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                # ctx_vectors = torch.empty(self.N, n_ctx, ctx_dim, dtype=dtype) 
                ctx_global = torch.empty(self.N, n_ctx, ctx_dim, dtype=dtype)
                ctx_local = torch.empty(self.N, n_ctx, ctx_dim, dtype=dtype) 
            
            # nn.init.normal_(ctx_vectors, std=0.02)   # define the prompt to be trained
            nn.init.normal_(ctx_global, std=0.02)   # define the prompt to be trained
            nn.init.normal_(ctx_local, std=0.02)   # define the prompt to be trained
            prompt_prefix = " ".join(["X"] * n_ctx)    

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        self.ctx_global = nn.Parameter(ctx_global)
        self.ctx_local = nn.Parameter(ctx_local)
        self.lraca = LowRankAsymmetricCrossAttention(
            dim=ctx_dim,
            rank=cfg.TRAINER.RESIDUAL_PEFT.LRACA_RANK,
            dtype=dtype
        )

        self.alpha_max = cfg.TRAINER.RESIDUAL_PEFT.ALPHA_MAX
        self.residual_tau = cfg.TRAINER.RESIDUAL_PEFT.TAU

        alpha_init = cfg.TRAINER.RESIDUAL_PEFT.ALPHA_INIT
        alpha_ratio = alpha_init / self.alpha_max

        self.alpha_logit = nn.Parameter(
            torch.tensor(
                [torch.logit(torch.tensor(alpha_ratio)).item()],
                dtype=torch.float32
            )
        )
        
        classnames = [name.replace("_", " ") for name in classnames]   
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])   
        tokenized_prompts = tokenized_prompts.repeat(self.N, 1) 

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) 

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.GL_SVDMSE.CLASS_TOKEN_POSITION

    def build_prompts(self, ctx):
        """
        ctx: [N, n_ctx, D]
        returns: [N * n_cls, prompt_length, D]
        """

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

        ctx_local_cond = torch.cat(
            [
                self.ctx_local[:, :1, :] + cross_anchor,
                self.ctx_local[:, 1:, :]
            ],
            dim=1
        )

        prompts_global = self.build_prompts(ctx_global)
        prompts_local = self.build_prompts(ctx_local_cond)

        alpha_i = self.alpha_max * torch.sigmoid(self.alpha_logit)
        alpha_i = alpha_i.to(dtype=ctx_local_cond.dtype)

        return prompts_global, prompts_local, alpha_i

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
        self.N = cfg.TRAINER.GL_SVDMSE.N

    def train(self, mode=True):
        super().train(mode)

        # 冻结 encoder 始终 eval，但不要 no_grad 包住 TextEncoder
        # 否则 Prompt 梯度会断
        self.image_encoder.eval()
        self.text_encoder.eval()

        return self


    def forward(self, image, idx=None, mode="personalized"):
        tokenized_prompts = self.tokenized_prompts

        prompts_global, prompts_local, alpha_i = self.prompt_learner()

        # 图像编码器冻结，可 no_grad
        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True
            )

        # 注意：TextEncoder 不可以 no_grad
        # Prompt 梯度要穿过冻结 TextEncoder 回到 ctx
        text_global = self.text_encoder(prompts_global, tokenized_prompts)
        text_global = text_global / text_global.norm(dim=-1, keepdim=True)

        text_local = self.text_encoder(prompts_local, tokenized_prompts)
        text_local = text_local / text_local.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()

        global_logits = logit_scale * image_features @ text_global.t()
        local_logits = logit_scale * image_features @ text_local.t()

        # Local 只学“相对 Global 的修正”
        raw_delta = local_logits - global_logits.detach()

        tau = self.prompt_learner.residual_tau
        bounded_delta = tau * torch.tanh(raw_delta.float() / tau)
        bounded_delta = bounded_delta.to(local_logits.dtype)

        personalized_logits = (
            global_logits.detach()
            + alpha_i * bounded_delta
        )

        if mode == "global":
            return global_logits

        if mode == "personalized":
            return personalized_logits

        if mode == "both":
            saturation = (
                raw_delta.float().abs() > tau
            ).float().mean()

            aux = {
                "alpha": alpha_i.detach().float().mean(),
                "delta_abs_mean": bounded_delta.detach().float().abs().mean(),
                "delta_saturation": saturation.detach()
            }

            return global_logits, personalized_logits, aux

        raise ValueError(f"Unknown mode: {mode}")

# @TRAINER_REGISTRY.register()
class GL_SVDMSE(TrainerX):
    """
    It is based on CoOp.
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.GL_SVDMSE.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        self.lambda_orthogonal = cfg.TRAINER.GL_SVDMSE.lambda_orthogonal
        self.alpha = cfg.TRAINER.GL_SVDMSE.alpha
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.GL_SVDMSE.PREC == "fp32" or cfg.TRAINER.GL_SVDMSE.PREC == "amp":
            # CLIP's default precision is fp16
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
            # device0 = torch.device("cuda:0")
            device1 = torch.device("cuda")
            self.model.to(self.device)
            self.model.text_encoder.to(device1)
            self.model.text_encoder=nn.DataParallel(self.model.text_encoder)
        else:
            self.model.to(self.device)
        
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.GL_SVDMSE.PREC == "amp" else None


    def forward_backward(self, batch_idx, batch, **kwargs):
        image, label = self.parse_batch_train(batch)

        global_logits, personalized_logits, aux = self.model(
            image, mode="both"
        )

        loss_global = F.cross_entropy(global_logits, label)
        loss_personal = F.cross_entropy(personalized_logits, label)

        # =========================================================
        # Debug only: 检查 Personal Loss 是否泄漏到 Global Prompt
        # 只在整个训练第一次执行，避免拖慢正式训练
        # =========================================================
        if not getattr(self, "_gradient_isolation_checked", False):
            self.optim.zero_grad(set_to_none=True)

            # 只反传个性化损失
            loss_personal.backward(retain_graph=True)

            global_grad = self.model.prompt_learner.ctx_global.grad

            global_grad_max = (
                0.0
                if global_grad is None
                else global_grad.detach().abs().max().item()
            )

            print(
                f"[Gradient Isolation Check] "
                f"max |dL_personal / dctx_global| = {global_grad_max:.3e}"
            )

            assert global_grad_max < 1e-10, (
                "Gradient leakage detected: loss_personal still updates "
                "prompt_learner.ctx_global. Check global_logits.detach(), "
                "ctx_global.detach(), and LRACA K/V paths."
            )

            # 清掉本次 debug 产生的梯度，再进入正常训练
            self.optim.zero_grad(set_to_none=True)
            self._gradient_isolation_checked = True

        # =========================================================
        # 正常训练
        # =========================================================
        lambda_p = self.cfg.TRAINER.RESIDUAL_PEFT.LAMBDA_P
        loss = loss_global + lambda_p * loss_personal

        self.model_backward_and_update(loss)

        return {
            "loss": loss.item(),
            "loss_global": loss_global.item(),
            "loss_personal": loss_personal.item(),
            "global_acc": compute_accuracy(global_logits, label)[0].item(),
            "personal_acc": compute_accuracy(personalized_logits, label)[0].item(),
            "alpha": float(aux["alpha"].item()),
            "delta_saturation": float(aux["delta_saturation"].item()),
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

        # By default, the best model is loaded
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

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)
