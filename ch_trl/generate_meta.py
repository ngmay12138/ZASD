import os
import json

class Meta:
    CLSNAMES = [
        "01", "02", "03"
    ]

    def __init__(self, root):
        self.root = root
        self.meta_path = f"{root}/meta.json"

    def run(self):
        info = {"train": {}, "test": {}}
        
        for cls_name in self.CLSNAMES:
            cls_dir = os.path.join(self.root, cls_name)
            for phase in ["train", "test"]:
                phase_info = []
                phase_dir = os.path.join(cls_dir, phase)
                
                for specie in os.listdir(phase_dir):
                    is_abnormal = specie != "good"
                    img_dir = os.path.join(phase_dir, specie)
                    img_names = sorted(os.listdir(img_dir))
                    
                    mask_dir = os.path.join(cls_dir, "ground_truth", specie)
                    mask_names = sorted(os.listdir(mask_dir)) if is_abnormal and os.path.exists(mask_dir) else []
                    
                    for idx, img_name in enumerate(img_names):
                        img_path = os.path.join(cls_name, phase, specie, img_name)
                        mask_path = os.path.join(cls_name, "ground_truth", specie, mask_names[idx]) if is_abnormal else ""
                        
                        phase_info.append({
                            "img_path": img_path,
                            "mask_path": mask_path,
                            "cls_name": cls_name,
                            "specie_name": specie,
                            "anomaly": int(is_abnormal)
                        })
                
                info[phase][cls_name] = phase_info
        
        with open(self.meta_path, "w") as f:
            json.dump(info, f, indent=4)


if __name__ == "__main__":
    runner = Meta(root="btad")
    runner.run()