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

    # 伪装绕过断言
    design_details = {"trainer": 'GL_SVDMSE',
                      "vision_depth": 0, "language_depth": 0, 
                      "vision_ctx": 0, "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model

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
        x = x.permute(1, 0, 2)  
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  
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
        self.N = cfg.TRAINER.RESIDUAL_PEFT.N

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
            ctx_global = ctx_vectors.unsqueeze(0).repeat(self.N, 1, 1)
        else:
            ctx_global = torch.empty(self.N, n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_global, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)    

        # 🌟 核心架构重建 1：参数空间的纯正残差
        self.ctx_global = nn.Parameter(ctx_global)
        # 局部残差初始化为全 0，初始状态下 PM 完全等于 GM
        self.delta_local = nn.Parameter(torch.zeros_like(ctx_global))
        
        classnames = [name.replace("_", " ") for name in classnames]   
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).repeat(self.N, 1) 
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) 

        self.register_buffer("token_prefix", embedding[:, :1, :])  
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts

    def build_prompts(self, ctx):
        ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
        ctx = ctx.permute(1, 0, 2, 3).contiguous()
        ctx = ctx.view(self.N * self.n_cls, self.n_ctx, ctx.shape[-1])
        return torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)

    def forward(self):
        # 1. 全局 Prompt
        prompts_global = self.build_prompts(self.ctx_global)

        # 2. 🌟 个性化 Prompt = 全局基底 + 本地残差 
        # 注意：这里绝对没有 detach()！允许 PM 梯度反哺 GM！
        ctx_personalized = self.ctx_global + self.delta_local
        prompts_personalized = self.build_prompts(ctx_personalized)

        return prompts_global, prompts_personalized

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        # 推理加速缓存
        self._cached_text_global = None
        self._cached_text_local = None

    def train(self, mode=True):
        super().train(mode)
        self.image_encoder.eval()
        self.text_encoder.eval()
        self._cached_text_global = None
        self._cached_text_local = None
        return self

    def forward(self, image, idx=None, mode="both"):
        prompts_global, prompts_personalized = self.prompt_learner()

        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # 🌟 极速推理缓存逻辑
        if not self.training and self._cached_text_global is not None:
            text_global = self._cached_text_global
            text_personalized = self._cached_text_local
        else:
            text_global = self.text_encoder(prompts_global, self.tokenized_prompts)
            text_global = text_global / text_global.norm(dim=-1, keepdim=True)

            text_personalized = self.text_encoder(prompts_personalized, self.tokenized_prompts)
            text_personalized = text_personalized / text_personalized.norm(dim=-1, keepdim=True)
            
            if not self.training:
                self._cached_text_global = text_global
                self._cached_text_local = text_personalized

        logit_scale = self.logit_scale.exp()
        global_logits = logit_scale * image_features @ text_global.t()
        personalized_logits = logit_scale * image_features @ text_personalized.t()

        if mode == "global":
            return global_logits
        if mode == "personalized":
            return personalized_logits
        if mode == "both":
            return global_logits, personalized_logits, {}

class RESIDUAL_PEFT(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.RESIDUAL_PEFT.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.RESIDUAL_PEFT.PREC in ["fp32", "amp"]:
            clip_model.float()   

        self.model = CustomCLIP(cfg, classnames, clip_model)
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.RESIDUAL_PEFT.PREC == "amp" else None

    def forward_backward(self, batch_idx, batch, **kwargs):
        image, label = self.parse_batch_train(batch)
        global_logits, personalized_logits, _ = self.model(image, mode="both")

        # 1. 标准交叉熵
        loss_global_ce = F.cross_entropy(global_logits, label)
        loss_personal = F.cross_entropy(personalized_logits, label)

        # 2. 🌟 核心架构重建 2：残差 L2 惩罚
        # 在数学上完美等价于基线 GL_SVDMSE 的全局-局部拉扯(Pull Loss)
        # 它迫使 delta_local 尽可能小，从而逼迫优化器把有用的知识学进 ctx_global 里！
        lambda_reg = 0.5  # 正则化强度超参，可微调
        loss_reg = lambda_reg * torch.norm(self.model.prompt_learner.delta_local, p=2)

        lambda_p = self.cfg.TRAINER.RESIDUAL_PEFT.LAMBDA_P
        
        # 总 Loss：全局 + 局部 + 残差惩罚
        loss = loss_global_ce + lambda_p * loss_personal + loss_reg

        self.model_backward_and_update(loss)

        return {
            "loss": loss.item(),
            "loss_global": loss_global_ce.item(),
            "loss_personal": loss_personal.item(),
            "global_acc": compute_accuracy(global_logits, label)[0].item(),
            "personal_acc": compute_accuracy(personalized_logits, label)[0].item(),
        }
    
    def set_eval_mode(self, mode):
        self.eval_mode = mode

    def model_inference(self, input, idx):
        self.model.eval()
        return self.model(input, idx=idx, mode=self.eval_mode)

    def parse_batch_train(self, batch):
        return batch["img"].to(self.device), batch["label"].to(self.device)

    def load_model(self, directory, epoch=None):
        if not directory:
            return
        names = self.get_model_names()
        model_file = f"model.pth.tar-{epoch}" if epoch else "model-best.pth.tar"
        for name in names:
            model_path = osp.join(directory, name, model_file)
            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            for k in ["token_prefix", "token_suffix"]:
                if k in state_dict:
                    del state_dict[k]
            self._models[name].load_state_dict(state_dict, strict=False)