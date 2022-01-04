import random
from transformers import MT5Tokenizer

tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")

def masking(input_ids, masked):
    EOS = 1
    ID = 250099
    c = 0
    prev_index = None
    for index in masked:
        if prev_index == index - 1:
            input_ids[index] = None
        else:
            input_ids[index] = ID - c
            c += 1
        prev_index = index
    return [ids for ids in input_ids if ids != None] + [EOS]

def masked(tokenizer, ratio=0.5, masking_source=masking, masking_target=masking):
    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
    buffers = {}
    def new_tokenizer(s, return_tensors="pt"):
        inputs = tokenizer(s, return_tensors=return_tensors)   # input のtensor 列

        if return_tensors != "pt":
            print(f'FIXME: return_tensors="{return_tensors}"')
            print(inputs)
            return inputs
        
        if s in buffers:
            target=buffers[s]
            inputs["input_ids"] = torch.tensor([target])
            inputs["attention_mask"] = inputs["input_ids"].new_ones(1, len(target))
            del buffers[s]
            return inputs
        
        input_ids = inputs["input_ids"].squeeze().tolist()

        if len(input_ids) == 1:
            return inputs
    
        input_ids = input_ids[:-1] # </s> を除いたinputs のlist
        n_tokens = len(input_ids)   # 字句数
        n = max(int((n_tokens / 2) * ratio), 1)
        input_masked = sorted(random.sample(list(range(0, n_tokens)), n))
        output_masked = list(set(list(range(0, n_tokens))) - set(input_masked))
        source = masking_source(input_ids[:], input_masked)
        target = masking_target(input_ids[:], output_masked)
        #print('source', source, tokenizer.decode(source))
        #print('target', target, tokenizer.decode(target))

        buffers[s] = target
        inputs["input_ids"] = torch.tensor([source])
        inputs["attention_mask"] = inputs["input_ids"].new_ones(1, len(source))
        return inputs
    return new_tokenizer