"""Offline reconstruction of the EXACT recorded GGUF source, with identity checks."""
import hashlib
import json
from pathlib import Path
import numpy as np
from static_quant.gemma import load_texts, sha256_file, input_group


class RecordedGGUFSource:
    def __init__(self, model_path, data_dir, manifest, threads=2):
        import torch
        import transformers
        import gguf
        from transformers import GemmaConfig, GemmaForCausalLM, GemmaTokenizerFast
        from transformers.integrations.ggml import GGUFGemmaConverter, GGUF_TOKENIZER_MAPPING, _gguf_parse_value
        self.torch=torch
        if sha256_file(model_path)!=manifest['gguf']['sha256']:
            raise ValueError('GGUF SHA256 differs from stored source')
        for key,actual in [('torch',torch.__version__),('transformers',transformers.__version__),('numpy',np.__version__)]:
            if actual!=manifest['library_versions'][key]:
                raise ValueError(f'{key} version differs: {actual} vs {manifest["library_versions"][key]}')
        torch.set_num_threads(threads)
        torch.manual_seed(manifest['seed'])
        torch.use_deterministic_algorithms(True)
        config=GemmaConfig(**manifest['config_source']['values'])
        config._attn_implementation='eager'
        with torch.device('meta'):
            self.model=GemmaForCausalLM(config)
        reader=gguf.GGUFReader(str(model_path))
        mapping={'token_embd.weight':'model.embed_tokens.weight','output_norm.weight':'model.norm.weight'}
        suffix={'attn_norm':'input_layernorm','ffn_norm':'post_attention_layernorm',
                'attn_q':'self_attn.q_proj','attn_k':'self_attn.k_proj','attn_v':'self_attn.v_proj',
                'attn_output':'self_attn.o_proj','ffn_gate':'mlp.gate_proj','ffn_up':'mlp.up_proj','ffn_down':'mlp.down_proj'}
        for layer in range(18):
            mapping.update({f'blk.{layer}.{a}.weight':f'model.layers.{layer}.{b}.weight' for a,b in suffix.items()})
        loaded=set()
        for t in reader.tensors:
            if t.name=='output.weight':
                continue
            if t.name not in mapping:
                raise ValueError(f'unrecognized GGUF tensor {t.name}')
            name=mapping[t.name]
            w=gguf.quants.dequantize(t.data,t.tensor_type).reshape(tuple(reversed(t.shape))).copy()
            if name.endswith(('layernorm.weight','norm.weight')):
                w-=1
            parent,_,leaf=name.rpartition('.')
            module=self.model.get_submodule(parent)
            if tuple(getattr(module,leaf).shape)!=w.shape:
                raise ValueError(f'shape mismatch {name}')
            setattr(module,leaf,torch.nn.Parameter(torch.from_numpy(w),requires_grad=False))
            loaded.add(name)
        self.model.tie_weights()
        if any(p.is_meta for p in self.model.parameters()):
            raise ValueError('unloaded model parameters')
        # Rotary frequencies are nonpersistent buffers and were created on meta.
        from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding
        self.model.model.rotary_emb=GemmaRotaryEmbedding(config,device='cpu')
        self.model.eval()
        tokenizer_data={}
        for key,target in GGUF_TOKENIZER_MAPPING['tokenizer'].items():
            field=reader.fields.get('tokenizer.'+key)
            if field is not None:
                values=[_gguf_parse_value(field.parts[i],field.types) for i in field.data]
                tokenizer_data[target]=values if field.types[0]==9 else values[0]
        converter=GGUFGemmaConverter(tokenizer_data)
        backend,kwargs=converter.converted(),converter.additional_kwargs
        self.tokenizer=GemmaTokenizerFast(tokenizer_object=backend,add_bos_token=True,add_eos_token=False,**kwargs)
        self.tokenizer.padding_side='right'
        self.datasets={}
        for split in ['calibration','validation']:
            info=manifest['datasets'][split];path=Path(data_dir)/f'{split}.jsonl'
            if sha256_file(path)!=info['file_sha256']:
                raise ValueError(f'dataset hash mismatch: {split}')
            texts,meta=load_texts(path,info['actual_samples'])
            if meta['selected_texts_sha256']!=info['selected_texts_sha256']:
                raise ValueError('selected texts differ')
            ids=[self.tokenizer(t,add_special_tokens=True,truncation=True,max_length=manifest['tokenizer']['sequence_length'])['input_ids'] for t in texts]
            digest=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            if digest!=info['token_ids_sha256']:
                raise ValueError(f'token IDs differ: {split}: {digest}')
            self.datasets[split]=ids
        self.modules=dict(self.model.named_modules())
        self.manifest=manifest

    def weight(self,name):
        return self.modules[name].weight.detach().numpy()

    def capture(self,layer,names,split,directory):
        """One original prefix per sample; capture unique groups, stop at last hook."""
        torch=self.torch;directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        tokens=sum(map(len,self.datasets[split]));groups={}
        for name in names:
            groups.setdefault(input_group(name),name)
        arrays={g:np.lib.format.open_memmap(directory/(g+'.npy'),mode='w+',dtype=np.float32,
                 shape=(tokens,self.modules[name].weight.shape[1])) for g,name in groups.items()}
        # Down-proj is last when present; otherwise select latest module in canonical order.
        order=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']
        stop_name=max(groups.values(),key=lambda n:order.index(n.split('.')[-1]))
        offsets={g:0 for g in groups}
        class StopCapture(Exception): pass
        handles=[]
        def hook_for(g,name):
            def hook(module,inputs):
                x=inputs[0].detach().cpu().numpy().reshape(-1,arrays[g].shape[1])
                if not np.isfinite(x).all(): raise ValueError('nonfinite activation')
                start=offsets[g];arrays[g][start:start+len(x)]=x;offsets[g]+=len(x)
                if name==stop_name: raise StopCapture()
            return hook
        for g,name in groups.items():
            handles.append(self.modules[name].register_forward_pre_hook(hook_for(g,name)))
        try:
            with torch.inference_mode():
                for ids in self.datasets[split]:
                    x=torch.tensor([ids],dtype=torch.int64)
                    try:self.model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False)
                    except StopCapture:pass
                    else:raise RuntimeError('capture stop hook did not run')
        finally:
            for h in handles:h.remove()
        if any(v!=tokens for v in offsets.values()):raise ValueError('capture token count mismatch')
        for a in arrays.values():a.flush()
        return arrays
