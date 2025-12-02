"""
Module to set up projection heads for both classification and summarization models.
Moved from train_model script.
"""

import torch
import os
from model.mixture_of_experts import MoEProjectionLayer, MixedMoEProjectionLayer
from transformers import PreTrainedModel, AutoConfig, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, BartForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput, SequenceClassifierOutput
from transformers.models.auto.configuration_auto  import AutoConfig
from torch import Tensor
from peft import PeftModel

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

class SeqClassWProjection(PreTrainedModel):
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
            num_experts=8,
            top_k=4
        )

        self.classifier = torch.nn.Linear(hidden_dim, num_labels) # classifier head

        self.pos_weight = (
            pos_weight.detach().clone().float() if pos_weight is not None else None
        )
        self.active_label_mask = (
            torch.tensor(active_label_mask, dtype=torch.float) if active_label_mask is not None else None
        )
    
    def save_custom(self, save_directory: str) -> None:
        """
        Custom save method to ensure projection head is saved along with base model.

        Args:
            save_directory (str): Directory to save the model components
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save base encoder
        self.base_encoder.base_model.base_model.save_pretrained(
            os.path.join(save_directory, "base_model")
        )

        # Save LoRA adapters
        self.base_encoder.save_pretrained(
            os.path.join(save_directory, "lora_adapters")
        )

        # Save MoE projection head
        torch.save(
            self.proj.state_dict(),
            os.path.join(save_directory, "moe_projection.pt")
        )

        # Save classifier head
        torch.save(
            self.classifier.state_dict(),
            os.path.join(save_directory, "classifier_head.pt")
        )

        # Save config
        self.config.save_pretrained(save_directory)
    
    @staticmethod
    def load_custom(save_directory: str) -> "SeqClassWProjection":
        """
        Custom load method to ensure projection head is loaded along with base model.

        Args:
            save_directory (str): Directory to load the model components from

        Returns:
            CodelessWrapper: Loaded model instance
        """
        # Load config
        config = AutoConfig.from_pretrained(save_directory)
        num_labels = config.num_labels

        # Load base encoder
        base_model = AutoModelForSequenceClassification.from_pretrained(
            config._name_or_path,
            config=config
        )

        # Load LoRA adapters
        peft_model = PeftModel.from_pretrained(
            base_model,
            os.path.join(save_directory, "lora_adapters")
        )

        # Initialize wrapper
        wrapper = SeqClassWProjection(
            config=config,
            base_encoder=peft_model,
            num_labels=num_labels
        )

        # Load MoE projection head
        moe_path = os.path.join(save_directory, "moe_projection.pt")
        wrapper.proj.load_state_dict(torch.load(moe_path, map_location="cpu"))

        # Load classifier head
        classifier_path = os.path.join(save_directory, "classifier_head.pt")
        wrapper.classifier.load_state_dict(torch.load(classifier_path, map_location="cpu"))

        return wrapper

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
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )


######### Summarization Wrappers #########
class Seq2SeqWProjection(BartForConditionalGeneration):
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

        kwargs.pop("num_items_in_batch", None)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("return_dict", None)
        kwargs.pop("use_cache", None)

        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        kwargs["use_cache"] = False

        if input_ids is not None:
            kwargs["input_ids"] = input_ids
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if decoder_input_ids is not None:
            kwargs["decoder_input_ids"] = decoder_input_ids
        if decoder_attention_mask is not None:
            kwargs["decoder_attention_mask"] = decoder_attention_mask
        if labels is not None:
            kwargs["labels"] = labels

        outputs = super().forward(**kwargs)

        if outputs.decoder_hidden_states is None:
            raise ValueError("Decoder hidden states are required for projection but were not returned.")

        hidden = outputs.decoder_hidden_states[-1]
        
        projected = self.proj(hidden)

        logits = self.lm_head(projected) + self.final_logits_bias

        loss = None
        if labels is not None:
            loss_function = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_function(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state
        )