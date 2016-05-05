#include "TEncSVM.h"


struct svm_model* TEncSVM::model;
std::string TEncSVM::modelPath;
bool TEncSVM::enableSVM;
struct svm_node* TEncSVM::svmNode;
double* TEncSVM::probEstimates;
int* TEncSVM::featIdx;
int TEncSVM::numFeatures;

void TEncSVM::init(){
    featIdx = (int*) malloc(sizeof(int)*MAX_FEAT); 
    model = svm_load_model(modelPath.c_str(), featIdx);
    numFeatures = featIdx[MAX_FEAT-1];
    svmNode = (struct svm_node *) malloc(numFeatures*sizeof(struct svm_node));
    probEstimates = (double*) malloc(sizeof(double)*2);

}

void TEncSVM::setCUFeatures(TComDataCU *&cu){

    
    
   // printf("Mode %d PU %d TrD %d\n", mode, puSize, trDepth);
    for(int i = 0; i < numFeatures; i++){
        svmNode[i].index = featIdx[i]; // C_MODE
        svmNode[i].value = getFeatureValue(featIdx[i], cu);
    }
   
    

    
}

bool TEncSVM::predictSplit(TComDataCU *&cu){
    
    setCUFeatures(cu);
    
    double predict_label = svm_predict_probability(model,svmNode,probEstimates);
    double not_split_prob = probEstimates[0];
    double split_prob = probEstimates[1];
        
    if (not_split_prob >= 0.5){
        return false;
    }
    if(split_prob > 0.5){
        return true;
    }
    
    return predict_label;

}

double TEncSVM::getFeatureValue(int idx, TComDataCU *&cu){
    double val = 0;
     
    int mvH = cu->getCUMvField(REF_PIC_LIST_0)->getMv( 0 ).getHor();
    int mvV = cu->getCUMvField(REF_PIC_LIST_0)->getMv( 0 ).getVer();
    double mvMod = (double) sqrt(mvH*mvH + mvV*mvV);
    
    double fracMvH = float(mvH % 4) / 4.0;
    double fracMvV = float(mvV % 4) / 4.0;
    double fracMvMod = (double) sqrt(fracMvH*fracMvH + fracMvV*fracMvV);

    switch(idx){
        case 0:
            return 0;
            break;
        case 1:
            return 0;
            break;
        case 6:
            if(cu->isSkipped(0)) val = 0;  // SKIP':'0', 'MERGE':'1', 'INTER':'2', 'INTRA': '3'}
            else if (cu->isInter(0) && !cu->getQtRootCbf( 0 )) val = 1;
            else if (cu->isInter(0)) val = 2;
            else val = 3;  
            break;
        case 9:
            val = (int) cu->getPartitionSize(0); // PU Size
            break;
        case 10:
            val = (int) cu->getTransformIdx(0); // TR DEPTH
            break;
        case 16:
            val = mvMod;        // INT MV MOD
            break;
        case 25:
            val = fracMvMod;  // FRAC MV MOD
            break;
        case 26:
            val = cu->getMVPIdx(RefPicList( REF_PIC_LIST_0 ), 0);  // MVP Idx
            break;
        default:
            fprintf(stderr, "Feature not supported!\n");
            exit(1);
            break;
    }
    return val;
}