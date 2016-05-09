#include "TEncSVM.h"
#include "TLibCommon/TComYuv.h"
#include "TLibCommon/TComPic.h"


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
   // double not_split_prob = probEstimates[0];
    double split_prob = probEstimates[1];
        
    if (split_prob >= 0.5){
        return true;
    }
    else{
        return false;
    }
    
    return (bool) predict_label;

}

double TEncSVM::getFeatureValue(int idx, TComDataCU *&cu){
    double val = 0;
    Pel *origSrc = cu->getPic()->getPicYuvOrg()->getAddr(COMPONENT_Y);
    int origStride = cu->getPic()->getPicYuvOrg()->getStride(COMPONENT_Y);
    
    Pel *resiSrc = cu->getPic()->getPicYuvResi()->getAddr(COMPONENT_Y);
    int resiStride = cu->getPic()->getPicYuvResi()->getStride(COMPONENT_Y);
    
    int fullMvH = cu->getCUMvField(REF_PIC_LIST_0)->getMv( 0 ).getHor();
    int fullMvV = cu->getCUMvField(REF_PIC_LIST_0)->getMv( 0 ).getVer();
   
    int mvH = fullMvH >> 2;
    int mvV = fullMvV >> 2;
    double mvMod = (double) sqrt(mvH*mvH + mvV*mvV);
    
    double fracMvH = float(fullMvH % 4) / 4.0;
    double fracMvV = float(fullMvV % 4) / 4.0;
    double fracMvMod = (double) sqrt(fracMvH*fracMvH + fracMvV*fracMvV);
    

    int fullPredMvH = cu->getCUMvField(RefPicList(REF_PIC_LIST_0))->getMvd( 0 ).getHor() + cu->getCUMvField(RefPicList(REF_PIC_LIST_0))->getMv( 0 ).getHor();
    int fullPredMvV = cu->getCUMvField(RefPicList(REF_PIC_LIST_0))->getMvd( 0 ).getVer() + cu->getCUMvField(RefPicList(REF_PIC_LIST_0))->getMv( 0 ).getVer();
    
    int predMvH = fullPredMvH >> 2;
    int predMvV = fullPredMvV >> 2;
    double predMvMod = (double) sqrt(predMvH*predMvH + predMvV*predMvV);
    
    double predFracMvH = float(fullPredMvH % 4) / 4.0;
    double predFracMvV = float(fullPredMvV % 4) / 4.0;
    double predFracMvMod = (double) sqrt(predFracMvH*predFracMvH + predFracMvV*predFracMvV);
    
    switch(idx){
        case 1:
            val = calculateAverage(cu, origSrc, origStride);
            break;
        case 2:
            val = calculateVariance(cu, origSrc, origStride);
            break;
        case 3:
            val = calculateAverage(cu, resiSrc, resiStride);
            break;
        case 4:
            val = calculateSAD(cu, resiSrc, resiStride);
            break;
        case 5:
            val = calculateVariance(cu, resiSrc, resiStride);
            break;
        case 6:
            if(cu->isSkipped(0)) val = 0;  // SKIP':'0', 'MERGE':'1', 'INTER':'2', 'INTRA': '3'}
            else if (cu->isInter(0) && !cu->getQtRootCbf( 0 )) val = 1;
            else if (cu->isInter(0)) val = 2;
            else val = 3;  
            break;
        case 7:
            val = cu->getTotalBits();
            break;
        case 8:
            val = cu->getTotalCost();
            break;
        case 9:
            val = (double) cu->getPartitionSize(0); // PU Size
            break;
        case 10:
            val = (double) cu->getTransformIdx(0); // TR DEPTH
            break;
        case 16:
            val = mvMod;        // INT MV MOD
            break;
        case 19:
            val = fracMvMod;
            break;
        case 22:
            val = predMvMod;
            break;
        case 25:
            val = predFracMvMod;  // FRAC MV MOD
            break;
        case 26:
            val = cu->getMVPIdx(RefPicList( REF_PIC_LIST_0 ), 0);  // MVP Idx
            break;
        default:
            fprintf(stderr, "Feature IDX %d not supported!\n", idx);
            exit(1);
            break;
    }
    return val;
}

double TEncSVM::calculateVariance(TComDataCU *&cu, Pel *src, int stride){
    int w = cu->getWidth(0);

    int diff;
    double avg = calculateAverage(cu, src, stride);
    double var = 0;
    for(int i = 0; i < w; i++){
        for(int j = 0; j < w; j++){
            diff = avg - src[j];
            var += diff*diff;
        }
        src+=stride;
    }
    
    return (var)/(w*w);
}

double TEncSVM::calculateAverage(TComDataCU *&cu, Pel *src, int stride){
    int w = cu->getWidth(0);
   
    double avg = 0;
    
    for(int i = 0; i < w; i++){
        for(int j = 0; j < w; j++){
            avg += src[j];
        }
        src+=stride;
    }
    
    return avg/(w*w);
}


double TEncSVM::calculateSAD(TComDataCU *&cu, Pel *src, int stride){
    int w = cu->getWidth(0);
   
    double avg = 0;
    
    for(int i = 0; i < w; i++){
        for(int j = 0; j < w; j++){
            avg += abs(src[j]);
        }
        src+=stride;
    }
    
    return avg/(w*w);
}

