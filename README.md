# Measuring Frailty by integrating behavioural and cardiovascular indicators through a deep-learning model
An frailty index (FI) is associated with mortality and vulnerability to adverse health outcomes. Current methods to calculate FI used uneasily accessible indicators, restricting its applications. 

By integrating BCs and CIs showing genetic causal effects on FI, a deep-learning model was developed to predict FI (BCDFI). The BCDFI is associated with the risks of multiple diseases and can perform GWAS analysis. 


<p align='center'>
<img src="./Figs/BCDFI.jpg" alt="architecture"/>
</p>


The code was built based on [AMFormer](https://github.com/aigc-apps/AMFormer/). Thanks a lot for their code sharing!
### Python version
3.7.9
### Dependencies
requirement.txt
### File description
1) Trained model parameters file: `./data/model/model.pth`
2) Input file: `./data/cate_feat.csv` and `./data/cont_feat.csv`
    The `cate_feat.csv` stores the input categorical variables and `cont_feat.csv` stores the input continuous variables, which meet the input requirements of AMFormer model.

### Testing
To run the code on the demo dataset, run:
`python -u main.py --save_dir ./result --gpu_id 7`


The prediction results are saved to the file `./result/result.csv` .



