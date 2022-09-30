# @Author: Khush Patel
# Dataset class for MLM and NSP pretraining tasks


from config import *


class MBDataset(torch.utils.data.Dataset):
    def __init__(self, rawdata, seed_value_changer=0):
        """Provide raw data list
        Args:
            Rawdata containing list 
        """
        self.rawdata = rawdata
        self.seed_value_changer = seed_value_changer

    def __len__(self):
        return len(self.rawdata)

    def __getitem__(self, idxx=int):
        """Actions done in this code block:
        1. Create input ids, token type ids, labels, attention mask
        2. Limit to max sequence length
        3. Add paddings if less than max sequence length 
        4. Add masking for MLM with 80% MASK, 10% random, 10% original.

        Returns:
            [dictionary]: [input_ids, token type ids, labels, attention mask]
        """
        # 1. Create input ids, token type ids, labels, attention mask
        # 2. Limit to max sequence length
        # 3. Add paddings if less than max sequence length

        torch.manual_seed(idxx + self.seed_value_changer)

        i = self.rawdata[idxx]
        input_ids = i[3][:max_position_embeddings]
        length = len(input_ids)
        padding = max_position_embeddings - length
        input_ids = input_ids + (padding*[0])
        labels = input_ids.copy()
        attention_mask = list(length * [1])
        attention_mask = attention_mask + (padding*[0])
        segment_type_ids = i[4][:max_position_embeddings]
        segment_type_ids = segment_type_ids + (padding*[0])
        los = max(i[1])
        if los >7:
            los=1
        else:
            los=0
        # Converting to torch.tensor
        labels = torch.tensor(labels)
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        segment_type_ids = torch.tensor(segment_type_ids)
        los = [los]
        los = torch.tensor(los)

        # 4. Add masking for MLM with 80% MASK, 10% random, 10% original.
        special_token_mask = np.zeros(max_position_embeddings)
        special_token_mask = torch.tensor(
            np.array(special_token_mask), dtype=torch.float)

        # https://pytorch.org/docs/stable/generated/torch.nonzero.html
        idx = max(torch.nonzero(input_ids))
        special_token_mask[:idx] = percent_tokens_to_mask
        special_token_mask[idx:] = 0

        # https://pytorch.org/docs/stable/generated/torch.bernoulli.html
        masked_indices = torch.bernoulli(special_token_mask).bool()
        # To prevent MCE loss calculation for unmasked tokens.
        labels[~masked_indices] = -100
        indices_replaced = torch.bernoulli(torch.full(
            labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = masked_token_encoding
        indices_random = torch.bernoulli(torch.full(
            labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced

        # https://pytorch.org/docs/stable/generated/torch.randint.html
        random_words = torch.randint(
            low=1, high=vocab_size, size=labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        return {"input_ids": input_ids, "segment_type_ids": segment_type_ids, "attention_mask": attention_mask, "labels": labels, "masked_indices": masked_indices, "next_sentence_label":los}
