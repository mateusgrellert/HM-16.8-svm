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
#include <cstdio>


class TEncSVM{
public:
    static struct svm_model* model;
    static std::string modelPath;
    static std::string rangePath;
    static bool enableSVM;
    static bool useScaling;
    static struct svm_node* svmNode;
    static double* probEstimates;

    static int* featIdx;
    static double* featMin;
    static double* featMax;
    static int numFeatures;

    static void init();
    static void parseRangeModel(std::string path);

    static bool predictSplit(TComDataCU *&cu, double time);
    static void setCUFeatures(TComDataCU *&cu, double time);
    static double scale(double value, int index);
    static void printSVMNode();

    static double getFeatureValue(int idx, TComDataCU *&cu, double time);
    static double calculateAverage(TComDataCU *&cu, Pel *src, int stride);
    static double calculateVariance(TComDataCU *&cu, Pel *src, int stride);
    static double calculateSAD(TComDataCU *&cu, Pel *src, int stride);


};

#endif	/* TENCSVM_H */

