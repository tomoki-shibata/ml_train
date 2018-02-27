//
//  main.cpp
//  google_test
//
//  Created by 柴田 智喜 on 2018/01/08.
//  Copyright © 2018年 柴田 智喜. All rights reserved.
//
/*
#include <iostream>

int main(int argc, const char * argv[]) {
    // insert code here...
    std::cout << "Hello, World!\n";
    return 0;
}
*/

#define GTEST_HAS_TR1_TUPLE 0
//#include "tree_tests.h"
//#include "clf_tree_test.h"
//#include "random_forest_test.h"
#include "light_gbm_test.h"
#include <random>
#include <iostream>
#include "gtest/gtest.h"
using namespace std;
int main(int argc, const char * argv[])
{
    /*
    Eigen::MatrixXd denomi,mat;
    denomi = Eigen::MatrixXd(3,2);
    mat = Eigen::MatrixXd(3,2);
    denomi<<
    1,2,
    2,3,
    3,6;
    
    mat <<
    2,2,
    3,3,
    2,2;
    

    //denomi = denomi.array().exp().matrix();
    denomi = (denomi.array() / mat.array()).matrix();
    
    for(int i= 0;i<=denomi.cols();++i){
        cout<<denomi(i,0)<<", "<<denomi(i,1)<<endl;
    }
    cout<<denomi.cols()<<endl;
    cout<<denomi.rows()<<endl;
    
    
    Eigen::MatrixXd A = Eigen::MatrixXd(3,3);
    A <<
    1,1,1,
    2,2,2,
    3,3,3;
    
    Eigen::VectorXd C = Eigen::VectorXd(3);
    C << 2,2,2;
    */
    
    ::testing::InitGoogleTest(&argc, (char **)argv);
    return RUN_ALL_TESTS();
}
