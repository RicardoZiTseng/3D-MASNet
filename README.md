# 3D-MASNet

This repository provides the experimental code for our paper "3D-MASNet: 3D Mixed-scale Asymmetric Convolutional Segmentation Network for 6-month-old Infant Brain MR Images".

Created by Zilong Zeng at Beijing Normal University. 
**For any questions, please contact 1941651789@qq.com or tengdazhao@bnu.edu.cn**

## Publication
If you find that this work is useful for your research, please consider citing our paper.

```
@article{zeng2021masnet,
  title={3D-MASNet: 3D Mixed-scale Asymmetric Convolutional Segmentation Network for 6-month-old Infant Brain MR Images},
  author={Zilong Zeng and Tengda Zhao and Lianglong Sun and Yihe Zhang and Mingrui Xia and Xuhong Liao and Jiaying Zhang and Dinggang Shen and Li Wang and Yong He},
  journal={Human Brain Mapping},
  year={2022}
}
```

The paper is avaliable at https://www.biorxiv.org/content/10.1101/2021.05.23.445294v1.


## Dataset
The dataset used for model training and validation is from [iSeg-2019](http://iseg2019.web.unc.edu/). The iSeg organizers provide 10 infant subjects with labels for model training, and 13 infant subjects without labels for model validation. Each subject consists of T1 and T2 images for segmentation.

### Model Fusion

Once your training process is over, you could configure your JSON file to fuse your model's parameters. An example of JSON file for the model fusion process shown in [deploy.json](/settings/deploy.json).
  - "task": This parameter need to set to `"deploy"`, which means that we're going to fuse the model's parameters.
  - "ori_model": List of pathes to models which are not fused.

Once you have configured your JSON file for model fusion, you need run this command like this:
```
python -m client.run ./settings/deploy.json
```

### Testing
Once your training or model fusion process is over, you can configure your JSON file to segment subjects' brain images. An example of JSON file for the model prediction shown in [predict.json](/settings/predict_undeploy.json).
  - "task": This parameter need to set to `"predict"`, which means that we're going to make model prediction.
  - "gpu_id": The id of the GPU which you want to use. For example, you want to use the second gpu, you should write `"1"`.
  - "save_folder": The path to the folder of the saved segmentation results.
  - "data_path": The path to the images to be segmented.
    - Notice!! If you want to use a different dataset here with T1 and T2 images, you dataset should be organized like this:
      ```
      ├── subject-1-label.hdr
      ├── subject-1-label.img
      ├── subject-1-T1.hdr
      ├── subject-1-T1.img
      ├── subject-1-T2.hdr
      ├── subject-1-T2.img
      ├── subject-2-label.hdr
      ├── subject-2-label.img
      ├── subject-2-T1.hdr
      ├── subject-2-T1.img
      ├── subject-2-T2.hdr
      ├── subject-2-T2.img
      ├── ...
      ```
  - "subjects": The list of ids of subjects to be segmented.
  - "predict_mode": two optional choice —— `"evaluation"` and `"prediction"`
    - `"evaluation"`: If you have labels, you can set this option and evaluate the model's accuracy.
    - `"prediction"`: If you do not have labels, you need to set this.
  - "model_files": The list of model files to be loaded for model prediction.
    - If there is only one model file's path in this parameter, the program will output one segmentation result predicted by this model file.
    - If there are multiple model files' pathes in this parameter, the program will adopt the majority voting strategy to combine these models' segmentation results.
  - "deploy": If the model file to be loaded has fused parameters, you should set this parameter as `true`; otherwise, you need to set here as `false`.

  We provide the pretrained model as example
  - if you have run the command list in [Model Fusion](#model-fusion), you could run command like this:
      ```
      python -m client.run ./settings/predict_example.json
      ```
  - We also provide the pretrained models used for the iSeg-2019 competition, if you want to obatin the same results as we provided to the iSeg organizers, you could run command like this:
      ```
      python -m client.run ./settings/predict_ensemble.json
      ```

  - We also release the quantitative evaluation results of iSeg-2019 competition: `evaluation_result_sw_bnu.xlsx`.
  - Our pilot study which names "3D-ACSNet" also released the pretrained models. You can check this project at https://github.com/RicardoZiTseng/3D-ACSNet. In this project, we adopted 11 models for ensemble and achieved slight worse results.

## Results
**Comparison of segmentation performance on the 13 validation infants of iSeg-2019 between the proposed method and the methods of the top 4 ranked teams.**
- DICE    
  |Team   |  CSF       | GM          | WM    | Average 
  |:----------:|:----------:|:-----------:|:-----:|:--------------:|
  |Brain_Tech|0.961|0.928|0.911|0.933|
  |FightAutism|0.960|0.929|0.911|0.933|
  |OxfordIBME|0.960|0.927|0.907|0.931|
  |QL111111|0.959|0.926|0.908|0.931|
  |Our|**0.961**|**0.931**|**0.912**|**0.935**|
- MHD
  |Team   |  CSF       | GM          | WM    | Average 
  |:----------:|:----------:|:-----------:|:-----:|:--------------:|
  |Brain_Tech|8.873|5.724|7.114|7.237|
  |FightAutism|9.233|5.678|**6.678**|7.196|
  |OxfordIBME|**8.560**|**5.495**|6.759|**6.938**|
  |QL111111|9.484|5.601|7.028|7.371|
  |Our|9.293|5.741|7.111|7.382|
- ASD
  |Team   |  CSF       | GM          | WM    | Average 
  |:----------:|:----------:|:-----------:|:-----:|:--------------:|
  |Brain_Tech|0.108|0.300|0.347|0.252|
  |FightAutism|0.110|0.300|0.341|0.250|
  |OxfordIBME|0.112|0.307|0.353|0.257|
  |QL111111|0.114|0.307|0.353|0.258|
  |Our|**0.107**|**0.292**|**0.332**|**0.244**|

