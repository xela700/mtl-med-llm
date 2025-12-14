from huggingface_hub import HfApi

api = HfApi()

repo_id = "xela700/biobert_large_ft_multilabel_clinical_classification"

def main():
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=True,
        exist_ok=True
    )

    api.upload_folder(
        repo_id=repo_id,
        folder_path="model/saved_model/huggingface_ft_biobert_large",
        commit_message="Initial commit of fine-tuned BioBERT large model for multi-label clinical classification w/ LoRA adapters + projection heads"
    )

if __name__ == "__main__":
    print("Creating repository and uploading model files...")
    main()