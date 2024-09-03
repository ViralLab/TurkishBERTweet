import{_ as l,o as c,c as d,p as f,e as u,a as e,h as b,f as v,b as t,w as n,d as s,g as y,F as x}from"./Df_DkY5G.js";import{_ as k}from"./D_qlBDOx.js";import{_ as w}from"./BgVM38eb.js";import"./BgZ1xh1b.js";import"./CU7MWXEm.js";const T={},p=o=>(f("data-v-32645570"),o=o(),u(),o),z={class:"header-background flex flex-col items-center justify-center space-y-8 bg-cover bg-center bg-no-repeat py-10 md:py-16 lg:py-24 text-gray-200"},E=p(()=>e("h1",{class:"text-6xl"},"Finetune TurkishBERTweet Using LoRA",-1)),R=p(()=>e("hr",{class:"w-1/12"},null,-1)),L=p(()=>e("p",{class:"text-2xl"},null,-1)),S=[E,R,L];function B(o,_){return c(),d("header",z,S)}const C=l(T,[["render",B],["__scopeId","data-v-32645570"]]),q=b("/images/data_csv_sample.png"),a=o=>(f("data-v-21afd141"),o=o(),u(),o),A={class:"inner pt-10"},F=a(()=>e("h3",null,"Finetuning TurkishBERTweet for Text Classification Using LoRA",-1)),I=a(()=>e("p",null,"In this section, we will guide you through the process of finetuning TurkishBERTweet on your own data form text Classification. ",-1)),j=a(()=>e("h3",null,"Setting Up the Environment for TurkishBERTweet",-1)),D=a(()=>e("br",null,null,-1)),N=a(()=>e("br",null,null,-1)),M=a(()=>e("h5",null,"1. Clone the Repository: Begin by cloning the TurkishBERTweet repository from GitHub.",-1)),P=a(()=>e("h5",null,"2. Navigate to the Directory: Move into the newly cloned directory.",-1)),$=a(()=>e("h5",null,"3. Set Up a Virtual Environment: Create a virtual environment to manage dependencies.",-1)),H=a(()=>e("h5",null,"4. Activate the Virtual Environment: Activate the virtual environment.",-1)),V=a(()=>e("h5",null,"5. Install Required Libraries: Install PyTorch and other essential libraries to run TurkishBERTweet. ",-1)),U=a(()=>e("h5",null,"6. Preparing your dataset for finetuning:",-1)),O=a(()=>e("p",null," Let's assume that you have a CSV file containing your samples with its corresponding labels. ",-1)),W=a(()=>e("div",{class:"border-border border p-2 rounded-md image_center"},[e("img",{src:q,alt:"Sample Image"})],-1)),G=a(()=>e("br",null,null,-1)),K=a(()=>e("p",null,[s(" I recommend converting your dataset into a HuggingFace dataset using "),e("a",{href:"https://huggingface.co/docs/datasets/en/index",class:"specialfont menu"},"datasets"),s(" library. To do so use the following script. Here I assume that you haven't split your data into train and test sets. ")],-1)),Q=a(()=>e("p",null,[s(" Now in your output directory, you will see "),e("span",{class:"specialfont"},"custom_ds"),s(" directory containing your data. ")],-1)),X=a(()=>e("h5",null,"7. It is time to finetune TurkishBERTweet with LoRA",-1)),J=a(()=>e("p",null,"First you need to prepare the config file in which you will set the training parameters and dataset paths, etc.",-1)),Y=a(()=>e("p",null,"Then you will run the following script passing the config file.",-1)),Z=a(()=>e("br",null,null,-1)),ee="git clone git@github.com:ViralLab/TurkishBERTweet.git",te="cd TurkishBERTweet",ae="python -m venv venv",oe=`# Need to mention that here Cuda 11.8 is being installed. Check whether your GPU Spec and install the compatible cuda.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install peft
pip install transformers
pip install datasets
pip install urlextract
pip install pandas
pip install -U scikit-learn
pip install joblib
pip install tqdm
`,ne=`import datasets as ds
ds.disable_caching()

from pathlib import Path
import joblib

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def create_dataset(dataset_path, output_path, train_size=0.8, ds_name="custom_ds"):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path)
    enc_target = preprocessing.LabelEncoder()
    y = enc_target.fit_transform(df["labels"]).tolist()
    x = df["sentence"].tolist()

    joblib.dump(enc_target, output_path / "meta.bin")
    df = pd.DataFrame({"sentence": x, "labels": y})

    df_train, df_test = train_test_split(
        df, train_size=train_size, stratify=df["labels"]
    )

    data_train = ds.Dataset.from_pandas(df_train)
    data_test = ds.Dataset.from_pandas(df_test)

    data_train = ds.DatasetDict(
        {
            "train": data_train,
            "test": data_test,
        }
    )
    data_train.save_to_disk(str(output_path / f"{ds_name}"))


data_path = r"path/data.csv"
output_path = r"path"
create_dataset(data_path, output_path, train_size=0.8, ds_name="custom_ds")
`,ie=`
{
    "model_name": "TurkishBERTweetLFT",
    "task_name": "TASK_NAME",
    "data_path": "DS_PATH",
    "pretrained_model_path": "VRLLab/TurkishBERTweet",
    "tokenizer_path": "VRLLab/TurkishBERTweet",
    "learning_rate": 3e-4,
    "weight_decay": 0.001,
    "epochs": 10,
    "train_batch_size": 32,
    "test_batch_size": 16,
    "early_stop_patience": 5,
    "early_stop_min_delta": 0.2,
    "input_len": 128,
    "task_type": "SEQ_CLS",
    "inference_mode": false,
    "target_modules": [
        "query",
        "value"
    ],
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1
}
`,re=`import sys, os
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import (
    get_peft_model,
    LoraConfig,
)

from datasets import load_from_disk, disable_caching
disable_caching()

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
)

from Preprocessor import preprocess


class History:
    def __init__(self) -> None:
        self.epoch_history = []

    def add_epoch_history(self, epoch_hist):
        self.epoch_history.append(epoch_hist)


def to_json(dictionary, path):
    with open(path, "w") as fh:
        json.dump(dictionary, fh, indent=4)


def save_classification_report(report, path):
    to_json(report, path)


class Metric:
    def __init__(self, name="train") -> None:
        self.name = name
        self.predictions = torch.tensor([])
        self.references = torch.tensor([])

        self.epoch_history = []

    def add_batch_history(self, references, predictions):
        self.predictions = torch.concat([self.predictions, predictions], dim=0)
        self.references = torch.concat([self.references, references], dim=0)

    def compute(self) -> dict:
        return {
            "weighted_f1": f1_score(
                self.references, self.predictions, average="weighted"
            ),
            "macro_f1": f1_score(self.references, self.predictions, average="macro"),
            "acc": accuracy_score(self.references, self.predictions),
            "weighted_percision": precision_score(
                self.references, self.predictions, average="weighted"
            ),
            "weighted_recall": recall_score(
                self.references, self.predictions, average="weighted"
            ),
        }

    def get_report(self):
        return classification_report(
            self.references, self.predictions, output_dict=True
        )

    def __repr__(self) -> str:
        return f"{self.name} metric: {self.compute()}"


def load_config(config_path):
    with open(config_path, "r") as fh:
        return json.load(fh)


def tokenize_function(tokenizer, examples, max_length):
    # max_length=None => use the model max length (it's actually the default)
    outputs = tokenizer(examples["sentence"], truncation=True, max_length=max_length)
    return outputs


def collate_fn(tokenizer, examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")


def train_and_get_best_model(
    config,
    model,
    device,
    train_dataloader,
    test_dataloader,
    optimizer,
    lr_scheduler,
):
    model = model.to(device)
    num_epochs = config["epochs"]
    best_f1 = -100
    best_model = None

    history = History()
    for epoch in range(num_epochs):
        epoch += 1
        # Training
        model = model.train()
        train_tqdm = tqdm(train_dataloader)
        train_tqdm.set_description(f"[Train] Epoch {epoch}")
        train_metric = Metric("train")
        for step, batch in enumerate(train_tqdm):
            batch.to(device)
            outputs = model(**batch)

            # setting num labels to avoid index out of range error
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            predictions = outputs.logits.argmax(dim=-1)
            references = batch["labels"]
            # Cache predictions and references for metric computation.
            train_metric.add_batch_history(
                references.cpu().detach(), predictions.cpu().detach()
            )

        # Evaluation
        eval_metric = Metric("eval")
        model = model.eval()
        eval_tqdm = tqdm(test_dataloader)
        eval_tqdm.set_description(f"[Eval] Epoch {epoch}")
        for step, batch in enumerate(eval_tqdm):
            batch.to(device)
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            references = batch["labels"]

            # Cache predictions and references for metric computation.
            eval_metric.add_batch_history(
                references.cpu().detach(), predictions.cpu().detach()
            )

        train_hist = train_metric.compute()
        eval_hist = eval_metric.compute()
        history.add_epoch_history({"train": train_hist, "eval": eval_hist})

        print("-" * 200)
        if eval_hist["weighted_f1"] > best_f1:
            best_f1 = eval_hist["weighted_f1"]
            best_model = model
            print("BEST MODEL UPDATED,", f"F1: {best_f1}")
        print(f"[TRAIN] Epoch {epoch}:", train_hist)
        print(f"[EVAL] Epoch {epoch}:", eval_hist)
        print("-" * 200)
    return best_model, history


def evaluate_best_model(model, device, data_loader, name="train"):
    model = model.to(device)
    model = model.eval()
    eval_tqdm = tqdm(data_loader)
    eval_tqdm.set_description(f"[{name.upper()}:BEST]")
    metric = Metric(name)
    for step, batch in enumerate(eval_tqdm):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        references = batch["labels"]

        # Cache predictions and references for metric computation.
        metric.add_batch_history(references.cpu().detach(), predictions.cpu().detach())

    return metric, metric.compute(), metric.get_report()


def get_experiment_folder_path(config):
    dataset_name = config["data_path"].split("/")[-1]
    experiment_folder = (
        f"experiments/{dataset_name}_{config['model_name']}_{config['task_name']}"
    )
    print("EXPERIMENT FOLDER:", experiment_folder)
    Path(experiment_folder).mkdir(exist_ok=True)
    return experiment_folder


def get_peft_config(config):
    peft_config = LoraConfig(
        task_type=config["task_type"],
        target_modules=config["target_modules"],
        r=config["r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
    )
    return peft_config


def init_model(config):
    peft_config = get_peft_config(config)
    model = AutoModelForSequenceClassification.from_pretrained(
        config["pretrained_model_path"],
        return_dict=True,
        num_labels=config["num_labels"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if any(k in config["pretrained_model_path"] for k in ("gpt", "opt", "bloom")):
        padding_side = "left"
    else:
        padding_side = "right"

    # loading Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["pretrained_model_path"], padding_side=padding_side
    )
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    experiment_folder_path = get_experiment_folder_path(config)

    # Lodading dataset
    my_dataset = load_from_disk(config["data_path"])
    num_labels = len(set(my_dataset["train"]["labels"]))
    config["num_labels"] = num_labels

    def apply_preprocess(example):
        example["sentence"] = apply_preprocess(example["sentence"].lower())
        return example

    my_dataset = my_dataset.map(apply_preprocess)
    my_dataset = my_dataset.filter(lambda example: example["sentence"].strip() != "")

    tokenized_datasets = my_dataset.map(
        lambda x: tokenize_function(tokenizer, x, config["input_len"]),
        batched=True,
        remove_columns=["sentence"],
    )

    # Instantiate dataloaders.
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        collate_fn=lambda x: collate_fn(tokenizer, x),
        batch_size=config["train_batch_size"],
    )
    test_dataloader = DataLoader(
        tokenized_datasets["test"],
        shuffle=False,
        collate_fn=lambda x: collate_fn(tokenizer, x),
        batch_size=config["test_batch_size"],
    )

    model = init_model(config)

    optimizer = AdamW(params=model.parameters(), lr=config["learning_rate"])

    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0.06 * (len(train_dataloader) * config["epochs"]),
        num_training_steps=(len(train_dataloader) * config["epochs"]),
    )

    best_model, history = train_and_get_best_model(
        config,
        model,
        device,
        train_dataloader,
        test_dataloader,
        optimizer,
        lr_scheduler,
    )

    # Save best model
    best_model.save_pretrained(experiment_folder_path + "/best_model")

    # Save history
    to_json(history.epoch_history, experiment_folder_path + "/history.json")

    train_metric, _, best_train_report = evaluate_best_model(
        best_model, device, train_dataloader, name="train"
    )
    test_metric, _, best_eval_report = evaluate_best_model(
        best_model, device, test_dataloader, name="eval"
    )

    save_classification_report(
        best_train_report, experiment_folder_path + "/best_train_report.json"
    )
    save_classification_report(
        best_eval_report, experiment_folder_path + "/best_test_report.json"
    )

    np.save(
        experiment_folder_path + "/best_train_predictions",
        train_metric.predictions.cpu().numpy(),
    )
    np.save(
        experiment_folder_path + "/best_test_predictions",
        test_metric.predictions.cpu().numpy(),
    )


if __name__ == "__main__":
    config = load_config(sys.argv[1])
    main(config)

`,se="python finetune.py path/config.json ",_e=v({__name:"FinetuneMain",setup(o){const _=String.raw`#Linux/Mac users
source venv/bin/activate
# Windows users
.\venv\Scripts\activate
`;return(m,h)=>{const i=k,r=y,g=w;return c(),d("main",A,[t(g,null,{default:n(()=>[e("div",null,[F,I,j,s(" To begin using the TurkishBERTweet model for sentiment analysis, you'll first need to set up your development environment. This involves cloning the TurkishBERTweet repository, creating a virtual environment, and installing the necessary libraries. Follow these steps to get started: "),D,N,M,t(r,null,{default:n(()=>[t(i,{code:ee,language:"bash",filename:"clone.sh"})]),_:1}),P,t(r,null,{default:n(()=>[t(i,{code:te,language:"bash",filename:"cd.sh"})]),_:1}),$,t(r,null,{default:n(()=>[t(i,{code:ae,language:"bash",filename:"venv.sh"})]),_:1}),H,t(r,null,{default:n(()=>[t(i,{code:_,language:"bash",filename:"activate.sh"})]),_:1}),V,t(r,null,{default:n(()=>[t(i,{code:oe,language:"text",filename:"install.sh"})]),_:1}),U,O,W,G,K,t(r,null,{default:n(()=>[t(i,{code:ne,language:"python",filename:"create_dataset.py"})]),_:1}),Q,X,J,t(r,null,{default:n(()=>[t(i,{code:ie,language:"json",filename:"config.json"})]),_:1}),Y,t(r,null,{default:n(()=>[t(i,{code:se,language:"bash",filename:"run_finetune.sh"})]),_:1}),t(r,null,{default:n(()=>[t(i,{code:re,language:"python",filename:"finetune.py"})]),_:1})]),Z]),_:1})])}}}),le=l(_e,[["__scopeId","data-v-21afd141"]]),ce={};function de(o,_){const m=C,h=le;return c(),d(x,null,[t(m),t(h)],64)}const ge=l(ce,[["render",de]]);export{ge as default};
