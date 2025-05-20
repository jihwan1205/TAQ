# TAQ
We provide source code to reproduce our project **$\rm{TAQ}$: Topology-Aware Quantization for GNNs** for CS471 Graph Machine Learning course at KAIST.

## 📦 Setting Up the Conda Environment

To reproduce the exact package setup used in this project, follow these steps:

1. **(Optional) Rename the environment**  
If you’d like a custom name, open `environment.yml` and change the top-level `name:` field.

2. **Create the environment**  
```bash
conda env create -f environment.yml
```   
3. **Activate the environment**  
```bash
conda activate <env-name>
```

## Bash script for TAQ
---GCN-Cora---
```bash
python node_level_1.py  --dataset_name Cora --model GCN --taq
```

---GCN-CiteSeer---
```bash
python node_level_1.py  --dataset_name CiteSeer --model GCN --taq
```

---GAT-Cora---
```bash
python gat_nc_lsb.py --dataset_name Cora --taq
```

---GAT-CiteSeer---
```bash
python gat_nc_lsb.py --dataset_name CiteSeer --taq
```

## Bash script for A2Q comparison 

---GCN-Cora---
```bash
python node_level_1.py --lr_quant_bit_fea 0.1 --lr_quant_scale_fea 0.04 --a_loss 2.5 --lr_quant_scale_weight 0.02 --lr_quant_scale_xw 0.008 --drop_out 0.35 --weight_decay 0.02 --dataset_name Cora --model GCN
```

---GCN-CiteSeer---
```bash
python node_level_1.py --lr_quant_bit_fea 0.1 --lr_quant_scale_fea 0.04 --a_loss 1.5 --lr_quant_scale_weight 0.008 --lr_quant_scale_xw 0.008 --drop_out 0.5 --weight_decay 0.015 --dataset_name CiteSeer --model GCN 
```

---GAT-Cora---
```bash
python gat_nc_lsb.py --weight_decay 1e-3 --lr 0.01 --lr_quant_scale_fea 0.05 --lr_quant_scale_weight 0.005 --lr_quant_scale_gat_fea 0.05 --lr_quant_scale_gat 0.005 --lr_quant_bit_fea 0.1 --a_loss 0.3 --drop_out 0.6 --drop_attn 0.6 --dataset_name Cora
```

---GAT-CiteSeer---
```bash
python gat_nc_lsb.py --weight_decay 1e-3 --lr 0.005 --lr_quant_scale_fea 0.05 --lr_quant_scale_weight 0.01 --lr_quant_scale_gat_fea 0.05 --lr_quant_scale_gat 0.005 --lr_quant_bit_fea 0.1 --a_loss 0.25 --drop_out 0.6 --drop_attn 0.6 --dataset_name CiteSeer
