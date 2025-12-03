import torch


pad_id = 0

class Dataset(torch.utils.data.Dataset):
    def __init__(self, loaded_data, max_len=512):
        self.data = []

        for inputs, outputs in zip(loaded_data['inputs'], loaded_data['outputs']):
            if len(inputs) > max_len:
                continue
            if len(inputs) + len(outputs) > max_len:
                outputs = outputs[: max_len - len(inputs)]

            self.data.append({
                'input_ids': torch.LongTensor(inputs + outputs).unsqueeze(0),   # during training, the whole sequence is inputs + outputs
                'labels': torch.LongTensor([-100]*len(inputs) + outputs).unsqueeze(0),  # loss is only calculated for outputs
                'attention_mask': torch.ones(1, len(inputs) + len(outputs)) # attention mask for padding tokens, 1 for not masked
            })
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def custom_collator(data_lst):
    # collate a list of element to a batch
    data_batch = {'input_ids': [], 'labels': [], 'attention_mask': []}
    # find the max length in this batch to pad to
    max_len = max([data['input_ids'].size(1) for data in data_lst])
    for data in data_lst:
        data_len = data['input_ids'].size(1)
        data_batch['input_ids'].append(
            # concat 1 for padding (1 is the pad_id for CodeGPT), pad_id may change for different models
            torch.cat([data['input_ids'], torch.ones(1, max_len - data_len).fill_(pad_id).long()], dim=1)
        )
        data_batch['labels'].append(
            # concat -100 for padding so they won't have loss. same for all models
            torch.cat([data['labels'], torch.zeros(1, max_len - data_len).fill_(-100).long()], dim=1)
        )
        data_batch['attention_mask'].append(
            # concat 0 for attention mask, 0 for masked. same for all models
            torch.cat([data['attention_mask'], torch.zeros(1, max_len - data_len)], dim=1)
        )
    data_batch['input_ids'] = torch.cat(data_batch['input_ids'], dim=0)
    data_batch['labels'] = torch.cat(data_batch['labels'], dim=0)
    data_batch['attention_mask'] = torch.cat(data_batch['attention_mask'], dim=0)
    return data_batch
