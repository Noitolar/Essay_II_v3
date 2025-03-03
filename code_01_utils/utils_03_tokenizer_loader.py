import transformers as tfm


def load_tokenizer(vocab_file_path: str)->tfm.BertTokenizerFast:
    tk = tfm.BertTokenizerFast(vocab_file=vocab_file_path)
    return tk


if __name__=="__main__":
    import datasets as ds
    import torch
    import os
    import utils_02_confg_loader as ut2
    import pprint

    os.chdir(r"C:\Users\MTX\Documents\Code\Python3\Essay_I_v2")
    dataset = ds.load_from_disk("data_00_cache/ShenzhenCDR_100k")
    tokenizer = tfm.BertTokenizerFast(vocab_file="data_00_cache/ShenzhenCDR_100k/vocab.txt")
    config = ut2.load_modernbert_json_config(
        json_file_path="code_00_configs/modernbert_config.json",
        vocab_size=tokenizer.vocab_size,
    )
    model = tfm.ModernBertForMaskedLM(config)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=3)
    for batch in loader:
        sentences = batch["sentence"]
        inputs = tokenizer(
            sentences,
            max_length=32,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False
        )
        pprint.pp(inputs)
        break