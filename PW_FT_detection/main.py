from ultralytics import YOLO, RTDETR
from utils import get_model_path
from munch import Munch
import yaml
import os

def main(config:str='./config.yaml'):

    # Load and set configurations from the YAML file
    with open(config) as f:
        cfg = Munch(yaml.load(f, Loader=yaml.FullLoader))

    if cfg.resume:  
        model_path = cfg.weights  
    else:  
        model_path = get_model_path(cfg.model_name)  
    
    if cfg.model == "YOLO":  
        model = YOLO(model_path)  
    elif cfg.model == "RTDETR":  
        model = RTDETR(model_path)  
    else:  
        raise ValueError("Model not supported")  

    with open(cfg.data) as f:
        data = yaml.safe_load(f)

    if not os.path.isabs(data["path"]):
        data["path"] = os.path.abspath(data["path"])
        with open(cfg.data, 'w') as f:
            yaml.dump(data, f)
    
    model.info()

    if cfg.task == "train":

        results = model.train(
            data=cfg.data,
            epochs=cfg.epochs, 
            imgsz=cfg.imgsz,
            device=cfg.device_train, 
            save_period=cfg.save_period, 
            workers=cfg.workers, 
            batch=cfg.batch_size_train, 
            val=cfg.val,
            project=f"runs/train_{cfg.exp_name}",
            name="exp",
            patience=cfg.patience,
            resume=cfg.resume
            )

    if cfg.task == "validation":

        metrics = model.val(
            data=cfg.data,
            save_json=cfg.save_json, 
            plots=cfg.plots, 
            device=cfg.device_val, 
            project=f'runs/val_{cfg.exp_name}', 
            name="exp",
            batch=cfg.batch_size_val) 

        metrics.box.map    # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps   # a list contains map50-95 of each category

    if cfg.task == "inference":

        results = model(cfg.test_data)
        save_path = os.path.join("inference_results", cfg.exp_name)
        os.makedirs(save_path, exist_ok=True)

        for i in range(len(results)):
            results[i][0].boxes
            results[i].save(filename=os.path.join(save_path,f"inference_{i}.jpg"))

if __name__ == "__main__":
    main()