import os
import logging
import warnings

from minigpt4.common.registry import registry
from minigpt4.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from minigpt4.datasets.datasets.cc_sbu_dataset import CCSBUDataset, CCSBUAlignDataset
from minigpt4.datasets.datasets.vqa_dataset import AOKVQADataset


@registry.register_builder("cc_sbu_align")
class CCSBUAlignBuilder(BaseDatasetBuilder):
    train_dataset_cls = CCSBUAlignDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/cc_sbu/align.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'filter_cap.json')],
            vis_root=os.path.join(storage_path, 'image'),
        )

        return datasets

@registry.register_builder("aokvqa")
class AOKVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = AOKVQADataset
    val_dataset_cls = AOKVQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/aokvqa/defaults.yaml",
    }
    
    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        anno_path=build_info.annotation
        vis_root=build_info.images
        coco_anno=build_info.coco
        
        if "sample_num" in self.config:
            sample_num = self.config.sample_num
        else:
            sample_num = None
        
        datasets = dict()
        
        # create datasets
        train_dataset_cls = self.train_dataset_cls
        datasets['train'] = train_dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            vis_root=vis_root,
            anno_path=anno_path,
            coco_anno=coco_anno,
            format=self.config.format,
            sample_num=sample_num,
        )
        datasets['train'][0]
        
        return datasets
    
@registry.register_builder("aokvqa_val")
class AOKVQAVALBuilder(BaseDatasetBuilder):
    val_dataset_cls = AOKVQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/aokvqa/defaults_val.yaml",
    }
    
    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        anno_path=build_info.annotation
        vis_root=build_info.images
        coco_anno=build_info.coco
        
        datasets = dict()
        
        val_dataset_cls = self.val_dataset_cls
        datasets['val'] = val_dataset_cls(
            vis_processor=self.vis_processors["eval"],
            text_processor=self.text_processors["eval"],
            vis_root=vis_root,
            anno_path=anno_path,
            coco_anno=coco_anno,
        )
        datasets['val'][0]
        
        return datasets
    
@registry.register_builder("gptvqa_round1")
class GPTVQARound1Builder(BaseDatasetBuilder):
    train_dataset_cls = AOKVQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/gptvqa_round1/defaults.yaml",
    }
    
    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        anno_path=build_info.train.annotation,
        vis_root=build_info.train.images
        coco_anno=build_info.train.coco
        
        if "sample_num" in self.config:
            sample_num = self.config.sample_num
        else:
            sample_num = None
        
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            vis_root=vis_root,
            anno_path=anno_path,
            coco_anno=coco_anno,
            format=self.config.format,
            sample_num=sample_num,
        )
        datasets['train'][0]

        return datasets
    
@registry.register_builder("gptvqa_round2")
class GPTVQARound2Builder(BaseDatasetBuilder):
    train_dataset_cls = AOKVQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/gptvqa_round2/defaults.yaml",
    }
    
    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        anno_path=build_info.train.annotation,
        vis_root=build_info.train.images
        coco_anno=build_info.train.coco
        
        if "sample_num" in self.config:
            sample_num = self.config.sample_num
        else:
            sample_num = None
        
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            vis_root=vis_root,
            anno_path=anno_path,
            coco_anno=coco_anno,
            format=self.config.format,
            sample_num=sample_num,
        )
        datasets['train'][0]

        return datasets