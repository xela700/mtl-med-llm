from utils.config_loader import load_config
from datasets import load_from_disk
from data.fetch_data import load_data

def check_labels():
    config = load_config()
    labels = config["data"]["task_2"]["data_path"]
    label_data = load_data(labels)
    label_ids = {code: i for i, code in enumerate(sorted(label_data["icd_code"].unique()))}

    print("Number of unique labels:", len(label_ids))
    print("Sample label_ids:", list(label_ids.items())[:5])
    print("Max index in label_ids:", max(label_ids.values()))

    data_path = config["data"]["task_1"]["tokenized_path"]
    dataset = load_from_disk(dataset_path=data_path)

    labels = dataset["labels"]
    lengths = [len(l) for l in labels]
    print(set(lengths))
    print(len(dataset))
    # # print(label_ids)

if __name__ == "__main__":
    check_labels()