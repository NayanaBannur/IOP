# IOP
TAT: Preprocessing a dataset
```python
python -m TAT.preprocessing --dataset CollegeMsg
```
TAT: Run the baseline model
```python
python -m TAT.main --dataset CollegeMsg --model TAT --gpu 0
```
TAT: Run the sequence prediction model
```python
python -m TAT.main_seq_pred --dataset CollegeMsg --model TAT --gpu 0
```
TAT: Run the prediction at timestep model
```python
python -m TAT.main_pred_t --dataset CollegeMsg --model TAT --gpu 0
```
JODIE: Generate embeddings
```python
python jodie.py --network CollegeMsg --model jodie --epochs 50
```
JODIE: IOP task (To use projected embeddings replace dyn_emb.py with dyn_emb_projected.py)
```python
python dyn_emb.py --network CollegeMsg --model jodie --epochs 50
```
See individual READMEs for more details.