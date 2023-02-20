## Training
### Environment
1. Install dependencies
    ```console
    pip install -r requirements.txt
    ```
2. Download data
    ```console
    bash scripts/download_data.sh
    ```
### Usage
```
python train.py --imgs_dir [] --masks_dir []
```
## Validate
```
python test.py --imgs_dir [] --masks_dir [] --model_name []
```
## Testing
### Usage
```
python demo.py --imgs_dir [] --result_dir [] --model_name []
```

- Reduce batch size to avoid CUDA out of memory.