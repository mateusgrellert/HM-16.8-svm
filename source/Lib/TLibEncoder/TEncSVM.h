/* 
 * File:   TEncSVM.h
 * Author: grellert
 *
 * Created on May 5, 2016, 10:32 AM
 */

#ifndef TENCSVM_H
#define	TENCSVM_H


#include "../Lib/TLibEncoder/svm.h"
#include "../Lib/TLibCommon/TComDataCU.h"
#include <string>
#include <cstdlib>
#include <cmath>


class TEncSVM{
public:
    static struct svm_model* model;
    static std::string modelPath;
    static bool enableSVM;
    static struct svm_node* svmNode;
    static double* probEstimates;

    static int* featIdx;
    static int numFeatures;

    static void init();
    static bool predictSplit(TComDataCU *&cu);
    static void setCUFeatures(TComDataCU *&cu);
    static double getFeatureValue(int idx, TComDataCU *&cu);
    static double calculateAverage(TComDataCU *&cu, Pel *src, int stride);
    static double calculateVariance(TComDataCU *&cu, Pel *src, int stride);
    static double calculateSAD(TComDataCU *&cu, Pel *src, int stride);


};

#endif	/* TENCSVM_H */

