import os
import json

class Meta:
    # CLSNAMES = [
    #     'audiojack','bottle_cap','button_battery','end_cap','eraser','fire_hood',
    #     'mint','mounts','pcb','phone_battery','plastic_nut','plastic_plug',
    #     'porcelain_doll','regulator','rolled_strip_base','sim_card_set','switch',
    #     'tape','terminalblock','toothbrush','toy','toy_brick','transistor1',
    #     'u_block','usb','usb_adaptor','vcpill','wooden_beads','woodstick','zipper'
    # ]
    CLSNAMES = [       'u_block','usb','usb_adaptor','vcpill','wooden_beads','woodstick','zipper'
    ]

    def __init__(self, root):
        self.root = root
        self.meta_path = f"{root}/meta.json"

    def run(self):
        info = {"train": {}, "test": {}}
        
        for cls_name in self.CLSNAMES:
            cls_dir = os.path.join(self.root, cls_name)
            if not os.path.exists(cls_dir):
                print(f"警告: 类别目录不存在: {cls_dir}")
                continue
                
            # 训练集为空
            info["train"][cls_name] = []
            
            # 处理测试集
            test_info = []
            
            # 处理OK文件夹 (good样本)
            ok_dir = os.path.join(cls_dir, "OK")
            if os.path.exists(ok_dir):
                # 获取所有样本文件夹 (S0001, S0002, ...)
                sample_folders = sorted([d for d in os.listdir(ok_dir) if os.path.isdir(os.path.join(ok_dir, d))])
                
                for sample_folder in sample_folders:
                    sample_path = os.path.join(ok_dir, sample_folder)
                    # 获取样本文件夹中的所有jpg图片
                    img_files = sorted([f for f in os.listdir(sample_path) if f.endswith('.jpg')])
                    
                    for img_file in img_files:
                        img_path = os.path.join(cls_name, "OK", sample_folder, img_file)
                        
                        test_info.append({
                            "img_path": img_path,
                            "mask_path": "",
                            "cls_name": cls_name,
                            "specie_name": "good",
                            "anomaly": 0
                        })
            
            # 处理NG文件夹 (异常样本)
            ng_dir = os.path.join(cls_dir, "NG")
            if os.path.exists(ng_dir):
                # 获取所有异常类型文件夹 (BX, ...)
                defect_types = sorted([d for d in os.listdir(ng_dir) if os.path.isdir(os.path.join(ng_dir, d))])
                
                for defect_type in defect_types:
                    defect_dir = os.path.join(ng_dir, defect_type)
                    # 获取异常类型下的所有样本文件夹
                    sample_folders = sorted([d for d in os.listdir(defect_dir) if os.path.isdir(os.path.join(defect_dir, d))])
                    
                    for sample_folder in sample_folders:
                        sample_path = os.path.join(defect_dir, sample_folder)
                        # 获取样本文件夹中的所有jpg图片（原图）
                        img_files = sorted([f for f in os.listdir(sample_path) if f.endswith('.jpg')])
                        
                        for img_file in img_files:
                            # 构建图片路径
                            img_path = os.path.join(cls_name, "NG", defect_type, sample_folder, img_file)
                            
                            # 构建对应的groundtruth图片路径
                            # 将.jpg替换为.png得到groundtruth文件名
                            mask_file = img_file.replace('.jpg', '.png')
                            mask_path = os.path.join(cls_name, "NG", defect_type, sample_folder, mask_file)
                            
                            # 检查groundtruth文件是否存在
                            full_mask_path = os.path.join(self.root, mask_path)
                            if not os.path.exists(full_mask_path):
                                mask_path = ""
                                print(f"警告: 未找到groundtruth文件: {full_mask_path}")
                            
                            test_info.append({
                                "img_path": img_path,
                                "mask_path": mask_path,
                                "cls_name": cls_name,
                                "specie_name": defect_type,  # 使用异常类型作为specie_name
                                "anomaly": 1
                            })
            
            info["test"][cls_name] = test_info
            
            print(f"处理完成: {cls_name}, 测试样本数: {len(test_info)}")
        
        # 写入meta.json文件
        with open(self.meta_path, "w") as f:
            json.dump(info, f, indent=4)
        
        print(f"meta.json已生成: {self.meta_path}")
        print(f"训练集: 所有类别均为空列表")
        print(f"测试集: 包含 {len(self.CLSNAMES)} 个类别")


if __name__ == "__main__":
    # 根据您的数据集路径进行修改
    runner = Meta(root="data/Real-IAD_2430")
    runner.run()