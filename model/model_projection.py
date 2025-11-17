"""
Module to set up projection heads for both classification and summarization models.
Moved from train_model script.
"""

import torch
from model.mixture_of_experts import MoEProjectionLayer, MixedMoEProjectionLayer
from transformers import PreTrainedModel, AutoConfig, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from torch import Tensor

######### Classification Wrappers #########

class CodeDescriptionWrapper(PreTrainedModel):
    """
    Wrapper designed for use with classification model training that incorporates the descriptions for each ICD-10 code.
    Uses a mostly-static encoder separate from trainable base encoder, and computes logits using dot product similarity.

    Attributes:
        base_encoder (AutoModel): base model used in classification training
        proj (nn.Sequential): sequence construction for the network
        pos_weight: positive class weights
        active_label_mask (Tensor): tensor of the mask for active labels (1.0 active; 0.0 inactive)
    """
    config_class = AutoConfig

    def __init__(self, config, base_encoder: AutoModelForSequenceClassification, label_embeds: Tensor, pos_weight: Tensor = None, active_label_mask: list[int] = None, proj_hidden: int = 256):
        super().__init__(config)
        self.base_encoder = base_encoder

        # Only doing MLP projection
        hidden_dim = label_embeds.size(1)
        # self.proj = torch.nn.Sequential( # Modified hidden dimension to stack more layers to see if that improves training
        #     torch.nn.Linear(hidden_dim, proj_hidden),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #     torch.nn.Linear(proj_hidden, proj_hidden),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #     torch.nn.Linear(proj_hidden, hidden_dim),
        #     torch.nn.LayerNorm(hidden_dim)
        # )

        self.proj = MoEProjectionLayer(hidden_dim, proj_hidden, num_experts=8, top_k=4, dropout=0.2)
        # End MLP projection

        self.register_buffer("label_embeds", label_embeds)
        self.pos_weight = pos_weight.detach().clone().float() if pos_weight is not None else None
        self.active_label_mask = torch.tensor(active_label_mask, dtype=torch.float) if active_label_mask is not None else None
    
    def forward(self, input_ids: list[list[int]] = None, attention_mask: list[list[int]] = None, token_type_ids: list[int] = None, labels: Tensor = None) -> dict[float:list[float]]:
        """
        Forward pass conducted during training modified for use with code description encoder running on the base model.
        Logits are reached by calculating the dot product similarity between the base encoder output and the code encoder.

        Args:
            input_ids (list[list[int]]): Input ids for the base encoder model
            attention_mask (list[list[int]]): Attention mask for the base encoder model
            token_type_ids (list[int]): Token type ids for the base encoder model
            labels (Tensor): Labels for the model

        Returns:
            dict[float:list[float]]: Loss and logits calculated during forward pass
        """
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        }
        # Outputs from trainable encoder model
        outputs = self.base_encoder.base_model.base_model(**model_inputs, output_hidden_states=True)

        note_embeds = outputs.last_hidden_state[:, 0, :]
        note_embeds = torch.nn.functional.normalize(note_embeds, dim=1)

        # Projection to align the embedding spaces of both base encoder and frozen encoder
        note_proj = self.proj(note_embeds)
        note_proj = torch.nn.functional.normalize(note_proj, dim=1)
        # End projection
        
        # Dot product logits between base encoder and frozen code desc label embeddings
        logits = torch.matmul(note_proj, self.label_embeds.T) # Fixed this - note_embeds -> note_proj

        loss = None
        if labels is not None:
            labels = labels.float().to(logits.device)
            loss_function = torch.nn.BCEWithLogitsLoss(
                pos_weight=self.pos_weight.to(logits.device) if self.pos_weight is not None else None,
                reduction="none"
            )
            loss_matrix = loss_function(logits, labels)

            if self.active_label_mask is not None:
                mask = self.active_label_mask.to(logits.device)
                loss_matrix = loss_matrix * mask
        
            loss = loss_matrix.mean()
        
        return {k: v for k, v in {"loss": loss, "logits": logits}.items() if v is not None}

class TrainableCodeDescriptionWrapper(PreTrainedModel):
    """
    Modified wrapper class for classification using note embeddings that also includes a trainable
    adapter to reshape frozen label embedding space in an attempt to overcome potential bottlenecks.
    """

    config_class = AutoConfig

    def __init__(
            self,
            config,
            base_encoder : AutoModelForSequenceClassification,
            label_embeds: Tensor,
            pos_weight: Tensor = None,
            active_label_mask: list[int] = None,
            proj_hidden: int = 256,
            adapter_init_scale: float = 0.01,
    ):
        super().__init__(config)
        self.base_encoder = base_encoder

        # Projection head
        hidden_dim = label_embeds.size(1)
        # self.proj = torch.nn.Sequential( # Modified hidden dimension to stack more layers to see if that improves training
        #     torch.nn.Linear(hidden_dim, proj_hidden),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #     torch.nn.Linear(proj_hidden, proj_hidden),
        #     torch.nn.GELU(),
        #     torch.nn.Dropout(0.2),
        #     torch.nn.Linear(proj_hidden, hidden_dim),
        #     torch.nn.LayerNorm(hidden_dim)
        # )
        self.proj = MoEProjectionLayer(hidden_dim, proj_hidden, num_experts=8, top_k=4, dropout=0.2)

        self.register_buffer("label_embeds", label_embeds) # label embeddings

        # Trainable adapter for labels
        self.label_adapter = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        torch.nn.init.eye_(self.label_adapter.weight) # starts at identity
        self.label_adapter.weight.data += adapter_init_scale * torch.randn_like(self.label_adapter.weight.data)
        # End new trainable adapter

        self.pos_weight = (
            pos_weight.detach().clone().float() if pos_weight is not None else None
        )
        self.active_label_mask = (
            torch.tensor(active_label_mask, dtype=torch.float) if active_label_mask is not None else None
        )

    def forward(self, input_ids: list[list[int]] = None, attention_mask: list[list[int]] = None, token_type_ids: list[int] = None, labels: Tensor = None) -> dict[float:list[float]]:
        """
        Forward pass conducted during training modified for use with code description encoder running on the base model.
        Logits are reached by calculating the dot product similarity between the base encoder output and the code encoder.

        Args:
            input_ids (list[list[int]]): Input ids for the base encoder model
            attention_mask (list[list[int]]): Attention mask for the base encoder model
            token_type_ids (list[int]): Token type ids for the base encoder model
            labels (Tensor): Labels for the model

        Returns:
            dict[float:list[float]]: Loss and logits calculated during forward pass
        """
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

        # Outputs from trainable encoder model
        outputs = self.base_encoder.base_model.base_model(**model_inputs, output_hidden_states=True)

        note_embeds = outputs.last_hidden_state[:, 0, :]
        note_embeds = torch.nn.functional.normalize(note_embeds, dim=1)

        # Projection to align the embedding spaces of both base encoder and frozen encoder
        note_proj = self.proj(note_embeds)
        note_proj = torch.nn.functional.normalize(note_proj, dim=1)
        # End projection

        # Adapter label embeddings
        adapted_labels = self.label_adapter(self.label_embeds)
        adapted_labels = torch.nn.functional.normalize(adapted_labels, dim=1)

        logits = torch.matmul(note_proj, adapted_labels.T)

        loss = None
        if labels is not None:
            labels = labels.float().to(logits.device)
            loss_function = torch.nn.BCEWithLogitsLoss(
                pos_weight=self.pos_weight.to(logits.device) if self.pos_weight is not None else None,
                reduction="none"
            )
            loss_matrix = loss_function(logits, labels)

            if self.active_label_mask is not None:
                mask = self.active_label_mask.to(logits.device)
                loss_matrix = loss_matrix * mask
        
            loss = loss_matrix.mean()
        
        return {k: v for k, v in {"loss": loss, "logits": logits}.items() if v is not None}

class CodelessWrapper(PreTrainedModel):
    """
    Modified wrapper for classification model that omits dot product similarity calculations with model outputs.
    Created to confirm if the code description embeddings are creating a bottleneck.
    Maintains projection head setup (either individual or MoE)
    """

    config_class = AutoConfig

    def __init__(
            self,
            config,
            base_encoder: AutoModelForSequenceClassification,
            num_labels: int,
            pos_weight: Tensor = None,
            active_label_mask: list[int] = None,
            proj_hidden: int = 256,
    ):
        super().__init__(config)
        self.base_encoder = base_encoder
        hidden_dim = base_encoder.config.hidden_size

        self.proj = MixedMoEProjectionLayer(
            input_dim=hidden_dim,
            hidden_dim=proj_hidden,
            num_experts=4,
            top_k=2
        )

        self.classifier = torch.nn.Linear(hidden_dim, num_labels) # classifier head

        self.pos_weight = (
            pos_weight.detach().clone().float() if pos_weight is not None else None
        )
        self.active_label_mask = (
            torch.tensor(active_label_mask, dtype=torch.float) if active_label_mask is not None else None
        )

    def forward(self, input_ids: list[list[int]] = None, attention_mask: list[list[int]] = None, token_type_ids: list[int] = None, labels: Tensor = None) -> dict[float:list[float]]:
        """
        Forward pass that incorporates projection head (singular or MoE).

        Args:
            input_ids (list[list[int]]): Input ids for the base encoder model
            attention_mask (list[list[int]]): Attention mask for the base encoder model
            token_type_ids (list[int]): Token type ids for the base encoder model
            labels (Tensor): Labels for the model

        Returns:
            dict[float:list[float]]: Loss and logits calculated during forward pass
        """
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

        # Outputs from trainable encoder model
        outputs = self.base_encoder.base_model.base_model(**model_inputs, output_hidden_states=True)

        note_embeds = outputs.last_hidden_state[:, 0, :]
        note_embeds = torch.nn.functional.normalize(note_embeds, dim=1)

        # Projection to align the embedding spaces of both base encoder and frozen encoder
        proj_embeds = self.proj(note_embeds)
        proj_embeds = torch.nn.functional.normalize(proj_embeds, dim=1)
        # End projection

        logits = self.classifier(proj_embeds)

        loss = None
        if labels is not None:
            labels = labels.float().to(logits.device)
            loss_function = torch.nn.BCEWithLogitsLoss(
                pos_weight=self.pos_weight.to(logits.device) if self.pos_weight is not None else None,
                reduction="none"
            )
            loss_matrix = loss_function(logits, labels)

            if self.active_label_mask is not None:
                mask = self.active_label_mask.to(logits.device)
                loss_matrix = loss_matrix * mask
        
            loss = loss_matrix.mean()
        
        return {k: v for k, v in {"loss": loss, "logits": logits}.items() if v is not None}


######### Summarization Wrappers #########

class Seq2SeqWProjection(AutoModelForSeq2SeqLM):
    """
    Class to add a small projection head to summarization model during training.
    """
    def __init__(self, config):
        super().__init__(config)
        hidden_size = config.d_model
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size // 2, hidden_size)
        )
    
    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        last_hidden = outputs.decoder_hidden_states[-1]
        projected = self.proj(last_hidden)

        logits = self.lm_head(projected) + self.model.final_logits_bias

        loss = None
        if labels is not None:
            loss_function = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_function(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return {"loss": loss, "logits": logits, "past_key_values": outputs.past_key_values}