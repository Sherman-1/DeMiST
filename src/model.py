import torch
import torch.nn as nn
from transformers import T5EncoderModel
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """
    See: https://einops.rocks/pytorch-examples.html

    #############################################################################
    # Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
    # https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
    # Copyright (c) 2024 Institut Pasteur                                       #
    #############################################################################
    """

    def __init__(self, temperature, attn_dropout=0.1, activation=nn.Softmax(dim=2)):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.activation = activation

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -torch.inf)

        attn = self.activation(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):

    """
    See: https://einops.rocks/pytorch-examples.html
    #############################################################################
    # Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
    # https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
    # Copyright (c) 2024 Institut Pasteur                                       #
    #############################################################################
    
    >>> bs = 4
    >>> d_model = 128
    >>> d_k = 16
    >>> d_v = 32
    >>> nq = 100
    >>> nv = 78
    >>> nhead = 8
    >>> mha = MultiHeadAttention(n_head=nhead, d_model=d_model, d_k=d_k, d_v=d_v)
    >>> q = torch.rand(bs, nq, d_model)
    >>> v = torch.rand(bs, nv, d_model)
    >>> k = torch.clone(v)
    >>> out, attn =  mha(q, k, v)
    >>> out.shape
    torch.Size([4, 100, 128])
    >>> attn.shape
    torch.Size([4, 100, 78])

    To compare when key_padding_mask is not None
    Put infinite values in k:
    >>> k[bs-1, nv-1] = torch.inf
    >>> k[bs-2, nv-2] = torch.inf
    >>> out, attn =  mha(q, k, v)

    Consequently the output contain nan at those positions
    >>> torch.isnan(out[bs-1, nv-1]).all()
    tensor(True)
    >>> torch.isnan(out[bs-2, nv-2]).all()
    tensor(True)

    and not at the other:
    >>> torch.isnan(out[bs-3, nv-3]).any()
    tensor(False)

    As well as the attention matrix (attn)
    >>> torch.isnan(attn[bs-1, :, nv-1]).all()
    tensor(True)
    >>> torch.isnan(attn[bs-2, :, nv-2]).all()
    tensor(True)
    >>> torch.isnan(attn[bs-3, :, nv-3]).any()
    tensor(False)

    Define a mask
    >>> key_padding_mask = torch.zeros(bs, nv, dtype=bool)
    >>> key_padding_mask[bs-1, nv-1] = True
    >>> key_padding_mask[bs-2, nv-2] = True
    >>> out, attn =  mha(q, k, v, key_padding_mask=key_padding_mask)
    >>> out.shape
    torch.Size([4, 100, 128])

    The output doesn't contain nan anymore as the infinite values are masked:
    >>> torch.isnan(out[bs-1, nv-1]).any()
    tensor(False)
    >>> torch.isnan(out[bs-2, nv-2]).any()
    tensor(False)

    The attn matrix contain 0 at the masked positions
    >>> attn.shape
    torch.Size([4, 100, 78])
    >>> (attn[bs-1, :, nv-1] == 0).all()
    tensor(True)
    >>> (attn[bs-2, :, nv-2] == 0).all()
    tensor(True)

    The attn matrix is softmaxed
    >>> mha = MultiHeadAttention(n_head=nhead, d_model=d_model, d_k=d_k, d_v=d_v, attn_dropout=0)
    >>> out, attn =  mha(q, k, v, key_padding_mask=key_padding_mask)
    >>> torch.isclose(attn.sum(dim=2), torch.ones_like(attn.sum(dim=2))).all()
    tensor(True)

    The user can define another activation function (or identity)
    >>> mha = MultiHeadAttention(n_head=nhead, d_model=d_model, d_k=d_k, d_v=d_v, attn_dropout=0, attn_activation=lambda x: x, attn_temperature=1.0)
    >>> out, attn =  mha(q, k, v, key_padding_mask=key_padding_mask)
    >>> torch.isclose(attn.sum(dim=2), torch.ones_like(attn.sum(dim=2))).all()
    tensor(False)

    User defined output dimension d_out:
    >>> q.shape
    torch.Size([4, 100, 128])
    >>> k.shape
    torch.Size([4, 78, 128])
    >>> v.shape
    torch.Size([4, 78, 128])
    >>> mha = MultiHeadAttention(n_head=nhead, d_model=d_model, d_k=d_k, d_v=d_v, d_out=64)
    >>> out, attn = mha(q, k, v, key_padding_mask=key_padding_mask)
    >>> out.shape, attn.shape
    (torch.Size([4, 100, 64]), torch.Size([4, 100, 78]))
    """

    def __init__(self, n_head:int, d_model:int, 
                 d_k:int=None, d_v:int=None, d_out:int=None, dropout:float=0.1, attn_dropout:float=0.1, 
                 attn_activation:callable=nn.Softmax(dim=2), attn_temperature=None, skip_connection=True):
        super().__init__()

        # Essaie 131224
        if not d_k : d_k = d_model
        if not d_v : d_v = d_model 

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.skip_connection = skip_connection

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        if attn_temperature is None:
            attn_temperature = np.power(d_k, 0.5)
        self.attention = ScaledDotProductAttention(temperature=attn_temperature, attn_dropout=attn_dropout, activation=attn_activation)

        if d_out is None:
            d_out = d_model
        elif d_out != d_model:
            self.skip_connection = False
        self.layer_norm = nn.LayerNorm(d_out)
        self.fc = nn.Linear(n_head * d_v, d_out)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None, key_padding_mask=None, average_attn_weights=True):
        
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        
        residual = q
        
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
        # Order for n_head and batch size: (n_head, sz_b, ...)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv
        
        if mask is not None:
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        if key_padding_mask is not None:  #(sz_b, len_k)
            key_padding_mask = torch.stack([key_padding_mask,]*n_head, dim=0).reshape(sz_b*n_head, 1, len_k) * torch.ones(sz_b*n_head, len_q, 1, dtype=torch.bool, device = q.device)
            if mask is not None:
                mask = mask + key_padding_mask
            else:
                mask = key_padding_mask

        output, attn = self.attention(q, k, v, mask=mask)
        # >>> output.shape
        # torch.Size([sz_b, len_q, d_v])
        # >>> attn.shape
        # torch.Size([sz_b*n_head, len_q, len_k])
        if average_attn_weights:
            attn = attn.view(n_head, sz_b, len_q, len_k)
            attn = attn.mean(dim=0)
        
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)
        
        output = self.dropout(self.fc(output))
        if self.skip_connection:
            output = self.layer_norm(output + residual)
        else:
            output = self.layer_norm(output)
        return output, attn

class MHAPooling(nn.Module):
    """
    MultiheadAttention Pooling (mhap)
    >>> nres = 35
    >>> ne = 1024
    >>> sb = 5
    >>> prottrans_embeddings = torch.rand(sb, nres, ne)
    >>> prottrans_embeddings.shape
    torch.Size([5, 35, 1024])
    >>> mhap = MHAP(embed_dim=ne)

    >>> attn_output, attn_output_weights = mhap(prottrans_embeddings)
    >>> attn_output.shape
    torch.Size([5, 1024])

    batch:
    >>> batch = torch.rand([64, 36, 1024])
    >>> batch.shape
    torch.Size([64, 36, 1024])
    >>> attn_output, attn_output_weights = mhap(batch)
    >>> attn_output.shape
    torch.Size([64, 1024])

    d_out:
    >>> mhap = MHAP(embed_dim=1024, d_out=128)
    >>> attn_output, attn_output_weights = mhap(batch)
    >>> attn_output.shape
    torch.Size([64, 128])
    """
    def __init__(self, embed_dim:int, d_out:int=None, n_head:int=8, dropout:float=0.1, 
                 attn_dropout:float = 0.1):
        super(MHAPooling, self).__init__()

        if d_out is None: 
            d_out = embed_dim

        self.attention = MultiHeadAttention(d_model=embed_dim, 
                                               n_head=n_head,  
                                               dropout=dropout, 
                                               attn_dropout=attn_dropout,
                                               d_out = d_out,
                                               d_v = d_out,
                                               d_k = d_out)
                                               
    def forward(self, embeddings, masks = None):
        
        Qi = embeddings.mean(axis=-2, keepdim=True) # Suppress mean pool for self attention
        attn_output, attn_output_weights = self.attention(q=Qi, k=embeddings, v=embeddings, key_padding_mask = masks)
        return torch.atleast_2d(torch.squeeze(attn_output)), attn_output_weights

class MLP(nn.Module):
    def __init__(self, input_dim=1024, num_classes=6, hidden_dim=512, dropout=0.1):

        """
        1024 -> 512 -> 256 -> num_classes
        """
        super().__init__()
        self.model = nn.Sequential(

            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, num_classes)
        )

        for module in self.model:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.model(x)

class ProtT5Classifier(nn.Module):
    """
    ProtT5 Encoder (with optional LoRA) + Mean Pooling + MLP Head.
    """
    def __init__(self, config, num_classes, loss_weights=None):
        super().__init__()
        
        model_name = config["model"]["checkpoint"]
        use_lora = config["model"].get("use_lora", False)

        print(f"Loading base model: {model_name}")
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        
        if use_lora:
            print("Applying LoRA to ProtT5 Encoder...")
            lora_config = LoraConfig(
                r=config["model"].get("lora_r", 8),
                lora_alpha=config["model"].get("lora_alpha", 32),
                target_modules=config["model"].get("lora_target_modules", ["q", "v"]),
                lora_dropout=config["model"].get("lora_dropout", 0.1),
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION 
            )
            self.encoder = get_peft_model(self.encoder, lora_config)
            self.encoder.print_trainable_parameters()
        else:
            print("Freezing ProtT5 base parameters (LoRA disabled)...")
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        self.classifier = MLP(input_dim=1024, hidden_dim=512, num_classes=num_classes)

        if loss_weights is not None:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=loss_weights)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        last_hidden_state = outputs.last_hidden_state 
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}